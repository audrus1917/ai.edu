# ai.edu

## AutoTokenizer и GPT2Tokenizer

`AutoTokenizer` - это не отдельный алгоритм токенизации, а фабрика, которая по имени модели
или по ее конфигурации выбирает подходящий конкретный класс токенизатора.

Для модели `gpt2` вызов `AutoTokenizer.from_pretrained("gpt2")` обычно приводит к
GPT-2-совместимому токенизатору. То есть по смыслу результат близок к явному использованию
`GPT2Tokenizer.from_pretrained("gpt2")`.

Главное различие такое:

- `AutoTokenizer` удобен, когда вы хотите менять модели без переписывания кода.
- `GPT2Tokenizer` удобен, когда вы точно знаете, что работаете только с GPT-2 и хотите явно это
	зафиксировать в коде.

С практической точки зрения отличие обычно не в качестве токенизации, а в уровне абстракции:

- `AutoTokenizer` = "подбери нужный класс сам".
- `GPT2Tokenizer` = "используй именно токенизатор GPT-2".

Еще один нюанс: `AutoTokenizer` может выбрать fast-вариант токенизатора, если он доступен,
поэтому он обычно более гибкий для прикладного кода.

Пример, который показывает реальный класс, возвращенный фабрикой:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.__class__.__name__)
```

Если нужно учебно показать механику именно GPT-2, можно использовать `GPT2Tokenizer` напрямую.
Если нужен более переносимый код под разные модели, обычно выбирают `AutoTokenizer`.

## Где локально лежит модель GPT-2

Если переменные `HF_HOME`, `TRANSFORMERS_CACHE` и `HUGGINGFACE_HUB_CACHE` не заданы,
Hugging Face сохраняет модель в стандартный кэш пользователя.

Для текущего окружения точный путь до snapshot-каталога `gpt2` такой:

```text
/home/andrus/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

Типичная структура внутри кэша такая:

- `refs` хранит указатели на revision, например `main`.
- `snapshots/<revision>` содержит файлы модели и токенизатора, которые читает библиотека.
- `blobs` содержит реальные данные по хешам.

У вас каталог `snapshots/...` состоит из симлинков на файлы в `blobs`.

Форматы файлов такие:

- `model.safetensors` - бинарные веса модели в формате Safetensors.
- `config.json` - JSON с архитектурой модели.
- `generation_config.json` - JSON с параметрами генерации по умолчанию.
- `vocab.json` - JSON-словарь токенов.
- `merges.txt` - текстовый файл с правилами BPE merge.
- `tokenizer.json` - сериализованный токенизатор целиком.
- `tokenizer_config.json` - JSON с настройками токенизатора.

То есть кратко:

- веса модели лежат в `model.safetensors`;
- конфигурация лежит в JSON-файлах;
- словарь токенизатора лежит в `vocab.json` и `merges.txt`.

Если нужно получить точный локальный путь прямо из Python, надежный способ такой:

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download("gpt2", local_files_only=True)
print(model_path)
```

В этом окружении `tokenizer.name_or_path` и `model.name_or_path` возвращают только `gpt2`,
а не путь до кэша, поэтому для точного пути лучше использовать именно `snapshot_download`.