# check_langchain

Минимальный pet-проект на Python для экспериментов с LangChain.

Что внутри:
- CLI-чат с историей диалога
- Поддержка 2 провайдеров: `ollama` (локально) и `openai`
- MVP RAG по локальным файлам `.md/.txt/.pdf` через PostgreSQL + pgvector
- Команды `/help`, `/clear`, `/reindex`, `/exit`

## 1) Подготовка

```bash
cd /home/andrus/la_strada/Apps/own/ai/check_langchain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Поднять PostgreSQL + pgvector

```bash
docker compose up -d
```

По умолчанию используется:
- DB: `langchain`
- User: `langchain`
- Password: `langchain`
- Port: `55433`

Добавь в `.env`:

```bash
DATABASE_URL=postgresql://langchain:langchain@localhost:55433/langchain
```

## 3) Инициализировать схему через Alembic

```bash
source .venv/bin/activate
alembic upgrade head
```

`upgrade head` применит:
- базовую схему таблиц,
- профили производительности `ivfflat` для размерностей эмбеддингов `768` и `1536`.

Если БД уже была создана вручную до Alembic, можно зафиксировать текущее состояние:

```bash
alembic stamp head
```

## 4) Вариант A: локально через Ollama (рекомендуется для старта)

1. Установи Ollama: https://ollama.com
2. Подтяни модель:

```bash
ollama pull llama3.1
```

3. Запусти чат:

```bash
python main.py --provider ollama --model llama3.1
```

## 5) Вариант B: через OpenAI API

1. Создай `.env` на основе `.env.example`
2. Добавь `OPENAI_API_KEY`
3. Запусти:

```bash
python main.py --provider openai --model gpt-4o-mini
```

## Полезные примеры

```bash
python main.py --provider ollama --model llama3.1 --temperature 0.0
python main.py --provider openai --model gpt-4o-mini --system "Отвечай кратко и по делу"
```

## 6) RAG по локальным документам (PostgreSQL)

1. Положи документы в папку `docs/` (поддерживаются `.md`, `.txt`, `.pdf`)
2. Запусти чат с RAG:

```bash
python main.py --provider ollama --model llama3.1 --rag --rag-dir docs
```

Для OpenAI:

```bash
python main.py --provider openai --model gpt-4o-mini --rag --rag-dir docs
```

Опциональные параметры RAG:
- `--embedding-model` (например, `nomic-embed-text` или `text-embedding-3-small`)
- `--rag-k` (сколько чанков подмешивать в контекст)
- `--chunk-size`, `--chunk-overlap` (параметры нарезки текста)
- `--db-url` (если не хочешь использовать `DATABASE_URL`)
- `--index-name` (логическое имя индекса в БД, по умолчанию `default`)
- `--session-id` (история сообщений в БД, по умолчанию `default`)
- `--explain-retrieval` (печатать профиль поиска: `ivfflat-768`, `ivfflat-1536` или `fallback`)

Во время чата можно вызвать `/reindex`, если ты добавил или изменил файлы в `docs/`.

## 7) Batch-индексация без запуска чата

```bash
python index_docs.py --provider ollama --rag-dir docs
```

Индексация сохранится в PostgreSQL (`index_name=default`).

Пример с кастомным индексом:

```bash
python index_docs.py --provider ollama --rag-dir docs --index-name my_docs
python main.py --provider ollama --model llama3.1 --rag --rag-dir docs --index-name my_docs --session-id chat1
```

## Идеи, как развить проект дальше

- Добавить RAG (чтение локальных документов)
- Подключить memory storage (SQLite)
- Сделать Telegram-бота на этой же логике
