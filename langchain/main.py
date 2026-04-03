"""Тествый скрипт для экспериментов с LangChain, Ollama и OpenAI.

Поддерживает чат с сохранением истории в PostgreSQL и RAG по локальным файлам
(форматы .md, .txt, .pdf). Для RAG используется расширение vector в PostgreSQL.

Пример запуска:
  python main.py --provider ollama --model llama3.1 --rag --db-url postgresql://user:pass@localhost/dbname
  python main.py --provider openai --model gpt-4o-mini --rag --db-url postgresql://user:pass@localhost/dbname

Команды в чате:
  /help   — показать помощь
  /clear  — очистить историю диалога
  /reindex — переиндексировать файлы для RAG
  /exit   — выйти.
"""

import argparse
import os
import uuid
from pathlib import Path
from typing import List

import psycopg

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pypdf import PdfReader


def build_llm(provider: str, model: str, temperature: float):
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def build_embeddings(provider: str, model: str):
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    if provider == "ollama":
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def to_pgvector(values: List[float]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def ensure_db_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("SELECT to_regclass('public.rag_chunks'), to_regclass('public.chat_messages')")
        rag_chunks, chat_messages = cur.fetchone()

    conn.commit()

    if rag_chunks is None or chat_messages is None:
        raise RuntimeError("Схема БД не инициализирована. Выполни: alembic upgrade head")


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size // 4, 0)

    chunks: List[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    text_len = len(cleaned_text)

    for start in range(0, text_len, step):
        end = min(start + chunk_size, text_len)
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break

    return chunks


def list_source_files(rag_dir: str) -> List[Path]:
    root = Path(rag_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Папка для RAG не найдена: {rag_dir}")

    supported_ext = {".md", ".txt", ".pdf"}
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in supported_ext]

    if not files:
        raise ValueError(f"В папке {rag_dir} нет поддерживаемых файлов (.md, .txt, .pdf)")

    return files


def load_documents(rag_dir: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    source_files = list_source_files(rag_dir=rag_dir)
    documents: List[Document] = []

    for path in source_files:
        print(f"Load document: {path}")
        if path.suffix.lower() == ".pdf":
            pdf_reader = PdfReader(str(path))
            raw_text = "\n".join((page.extract_text() or "") for page in pdf_reader.pages)
        else:
            try:
                raw_text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw_text = path.read_text(encoding="utf-8", errors="ignore")

        for idx, chunk in enumerate(
            chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": str(path), "chunk": idx},
                )
            )

    if not documents:
        raise ValueError(f"В файлах {rag_dir} не удалось извлечь текст для индексации")

    return documents


def index_documents_in_postgres(
    db_url: str,
    index_name: str,
    provider: str,
    embedding_model: str,
    rag_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    docs = load_documents(rag_dir=rag_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = build_embeddings(provider=provider, model=embedding_model)
    vectors = embeddings.embed_documents([doc.page_content for doc in docs])

    with psycopg.connect(db_url) as conn:
        ensure_db_schema(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rag_chunks WHERE index_name = %s", (index_name,))
            for doc, vector in zip(docs, vectors, strict=False):
                cur.execute(
                    """
                    INSERT INTO rag_chunks (index_name, source_path, chunk_no, content, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (
                        index_name,
                        str(doc.metadata.get("source", "unknown")),
                        int(doc.metadata.get("chunk", 0)),
                        doc.page_content,
                        to_pgvector(vector),
                    ),
                )
        conn.commit()

    return len(docs)


def retrieve_relevant_docs(
    db_url: str,
    index_name: str,
    query_text: str,
    rag_k: int,
    provider: str,
    embedding_model: str,
) -> tuple[List[Document], str, int]:
    embeddings = build_embeddings(provider=provider, model=embedding_model)
    query_vector = embeddings.embed_query(query_text)
    vector_dim = len(query_vector)
    retrieval_profile = "fallback"

    with psycopg.connect(db_url) as conn:
        ensure_db_schema(conn)
        with conn.cursor() as cur:
            if vector_dim in {768, 1536}:
                retrieval_profile = f"ivfflat-{vector_dim}"
                cur.execute(
                    f"""
                    SELECT source_path, chunk_no, content
                    FROM rag_chunks
                    WHERE index_name = %s AND vector_dims(embedding) = {vector_dim}
                    ORDER BY (embedding::vector({vector_dim})) <=> (%s::vector({vector_dim}))
                    LIMIT %s
                    """,
                    (index_name, to_pgvector(query_vector), rag_k),
                )
            else:
                cur.execute(
                    """
                    SELECT source_path, chunk_no, content
                    FROM rag_chunks
                    WHERE index_name = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (index_name, to_pgvector(query_vector), rag_k),
                )
            rows = cur.fetchall()

    docs = [Document(page_content=row[2], metadata={"source": row[0], "chunk": row[1]}) for row in rows]
    return docs, retrieval_profile, vector_dim


def load_chat_history(db_url: str, session_id: str) -> List[BaseMessage]:
    with psycopg.connect(db_url) as conn:
        ensure_db_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY id
                """,
                (session_id,),
            )
            rows = cur.fetchall()

    history: List[BaseMessage] = []
    for role, content in rows:
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def save_chat_message(db_url: str, session_id: str, role: str, content: str) -> None:
    with psycopg.connect(db_url) as conn:
        ensure_db_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (session_id, role, content),
            )
        conn.commit()


def print_help() -> None:
    print("\nКоманды:")
    print("  /help   — показать помощь")
    print("  /clear  — очистить историю диалога")
    print("  /reindex — переиндексировать файлы для RAG")
    print("  /exit   — выйти\n")


def chat_loop(
    system_prompt: str,
    provider: str,
    model: str,
    temperature: float,
    rag: bool,
    rag_dir: str,
    embedding_model: str,
    rag_k: int,
    chunk_size: int,
    chunk_overlap: int,
    db_url: str,
    session_id: str,
    index_name: str,
    explain_retrieval: bool,
) -> None:
    llm = build_llm(provider=provider, model=model, temperature=temperature)

    history: List[BaseMessage] = []
    if system_prompt:
        history.append(SystemMessage(content=system_prompt))

    if db_url:
        persisted = load_chat_history(db_url=db_url, session_id=session_id)
        history.extend(persisted)
        if persisted:
            print(f"Загружена история из PostgreSQL: сообщений {len(persisted)}")

    if rag:
        if not db_url:
            raise ValueError("Для режима --rag нужен --db-url или переменная DATABASE_URL")
        chunks_count = index_documents_in_postgres(
            db_url=db_url,
            index_name=index_name,
            provider=provider,
            embedding_model=embedding_model,
            rag_dir=rag_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"RAG включен: индекс в PostgreSQL обновлен, чанков: {chunks_count}")

    print("LangChain demo chat запущен.")
    print("Введите сообщение. /help для команд.\n")

    while True:
        try:
            user_text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return

        if not user_text:
            continue

        if user_text == "/exit":
            print("Выход.")
            return

        if user_text == "/help":
            print_help()
            continue

        if user_text == "/clear":
            history = [SystemMessage(content=system_prompt)] if system_prompt else []
            print("История очищена.")
            continue

        if user_text == "/reindex":
            if not rag:
                print("RAG не включен. Запусти с флагом --rag.")
                continue
            try:
                chunks_count = index_documents_in_postgres(
                    db_url=db_url,
                    index_name=index_name,
                    provider=provider,
                    embedding_model=embedding_model,
                    rag_dir=rag_dir,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                print(f"Готово. Переиндексировано чанков: {chunks_count}")
            except Exception as exc:
                print(f"Не удалось переиндексировать: {exc}")
            continue

        history.append(HumanMessage(content=user_text))
        if db_url:
            save_chat_message(db_url=db_url, session_id=session_id, role="user", content=user_text)

        try:
            messages_for_model: List[BaseMessage] = list(history)

            if rag:
                docs, retrieval_profile, vector_dim = retrieve_relevant_docs(
                    db_url=db_url,
                    index_name=index_name,
                    query_text=user_text,
                    rag_k=rag_k,
                    provider=provider,
                    embedding_model=embedding_model,
                )
                if explain_retrieval:
                    print(f"[retrieval] профиль={retrieval_profile}, размерность={vector_dim}, чанков={len(docs)}")
                context = "\n\n".join(
                    f"Источник: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
                    for doc in docs
                )
                rag_instruction = (
                    "Используй только контекст ниже для ответа. "
                    "Если контекста недостаточно, честно скажи об этом.\n\n"
                    f"Контекст:\n{context}"
                )
                messages_for_model.append(SystemMessage(content=rag_instruction))

            response = llm.invoke(messages_for_model)
        except Exception as exc:
            print(f"Ошибка вызова модели: {exc}")
            history.pop()
            continue

        ai_text = response.content if isinstance(response.content, str) else str(response.content)
        history.append(AIMessage(content=ai_text))
        if db_url:
            save_chat_message(db_url=db_url, session_id=session_id, role="assistant", content=ai_text)

        print(f"Ассистент: {ai_text}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pet-проект: чат с LangChain")
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="Провайдер LLM (по умолчанию: ollama)",
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Имя модели (например, llama3.1 для Ollama или gpt-4o-mini для OpenAI)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Температура генерации (по умолчанию: 0.2)",
    )
    parser.add_argument(
        "--system",
        default="Ты полезный ассистент для экспериментов с LangChain.",
        help="Системный промпт",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Включить RAG по локальным файлам",
    )
    parser.add_argument(
        "--rag-dir",
        default="docs",
        help="Папка с файлами .md/.txt/.pdf для RAG (по умолчанию: docs)",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Модель эмбеддингов (если пусто, берется по провайдеру)",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=4,
        help="Сколько чанков подтягивать в контекст (по умолчанию: 4)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Размер чанка для индексации (по умолчанию: 1200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Перекрытие чанков (по умолчанию: 200)",
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="Строка подключения PostgreSQL (если пусто, берется из DATABASE_URL)",
    )
    parser.add_argument(
        "--session-id",
        default="default",
        help="Идентификатор сессии чата в БД (по умолчанию: default)",
    )
    parser.add_argument(
        "--index-name",
        default="default",
        help="Имя RAG-индекса в PostgreSQL (по умолчанию: default)",
    )
    parser.add_argument(
        "--explain-retrieval",
        action="store_true",
        help="Показывать профиль retrieval (ivfflat/fallback) для каждого RAG-запроса",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    embedding_model = args.embedding_model
    if not embedding_model:
        embedding_model = "text-embedding-3-small" if args.provider == "openai" else "nomic-embed-text"

    db_url = args.db_url or os.getenv("DATABASE_URL", "")

    session_id = args.session_id.strip() or str(uuid.uuid4())

    try:
        chat_loop(
            system_prompt=args.system,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            rag=args.rag,
            rag_dir=args.rag_dir,
            embedding_model=embedding_model,
            rag_k=args.rag_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            db_url=db_url,
            session_id=session_id,
            index_name=args.index_name,
            explain_retrieval=args.explain_retrieval,
        )
    except ValueError as exc:
        print(exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
