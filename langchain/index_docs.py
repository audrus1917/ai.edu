import argparse
import os

from dotenv import load_dotenv

from main import index_documents_in_postgres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-индексация документов для RAG")
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="Провайдер эмбеддингов",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Модель эмбеддингов (если пусто, берется по провайдеру)",
    )
    parser.add_argument(
        "--rag-dir",
        default="docs",
        help="Папка с файлами .md/.txt/.pdf для RAG (по умолчанию: docs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Размер чанка",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Перекрытие чанков",
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="Строка подключения PostgreSQL (если пусто, берется из DATABASE_URL)",
    )
    parser.add_argument(
        "--index-name",
        default="default",
        help="Имя RAG-индекса в PostgreSQL (по умолчанию: default)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    embedding_model = args.embedding_model
    if not embedding_model:
        embedding_model = "text-embedding-3-small" if args.provider == "openai" else "nomic-embed-text"

    db_url = args.db_url or os.getenv("DATABASE_URL", "")
    if not db_url:
        print("Ошибка индексации: укажи --db-url или переменную окружения DATABASE_URL")
        return 1

    try:
        chunks_count = index_documents_in_postgres(
            db_url=db_url,
            index_name=args.index_name,
            provider=args.provider,
            embedding_model=embedding_model,
            rag_dir=args.rag_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as exc:
        print(f"Ошибка индексации: {exc}")
        return 1

    print(f"Индекс обновлен. Чанков в индексе: {chunks_count}")
    print(f"Индекс в PostgreSQL: {args.index_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
