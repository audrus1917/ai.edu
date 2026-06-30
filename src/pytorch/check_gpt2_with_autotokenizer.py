"""
* Тензоры: tokenizer.encode превращает ваши слова в векторы (тензоры), которые PyTorch может
  обрабатывать на видеокарте или процессоре.
* Веса модели: GPT2LMHeadModel подгружает миллионы параметров (чисел), которые были подобраны в
  процессе обучения на гигантских массивах текста.
* Forward Pass: Когда вызывается model.generate, PyTorch прогоняет ваши данные через слои нейросети
  (матричное умножение), чтобы предсказать следующее наиболее вероятное слово.

Важный нюанс: Оригинальная GPT-2 хорошо понимает английский. Если вам нужен пример на русском языке,
достаточно просто заменить название модели во второй строке на ai-forever/rugpt3small_based_on_gpt2.

"""

from huggingface_hub import snapshot_download

from transformers import AutoTokenizer, AutoModelForCausalLM


HF_TOKEN = "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"  # Ваш токен Hugging Face
os.environ["HF_TOKEN"] = HF_TOKEN 


def main() -> None:

    # 1. Загружаем предобученную модель и токенизатор (инструмент для перевода
    # текста в числа)
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_path = snapshot_download(model_name, local_files_only=True)

    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Local model path: {model_path}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    prompt = "PyTorch is amazing because"

    # 2. Передаем padding и возвращаем тензоры для нужного фреймворка (pt - PyTorch)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # 3. Передаем ВСЕ сгенерированные аргументы в модель (включая attention_mask)
    outputs = model.generate(
        **inputs, 
        max_length=50,
        pad_token_id=tokenizer.pad_token_id # Явно указываем ID токена заполнения
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

