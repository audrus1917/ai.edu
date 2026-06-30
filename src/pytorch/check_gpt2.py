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

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main() -> None:

    # 1. Загружаем предобученную модель и токенизатор (инструмент для перевода
    # текста в числа)
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 2. Исходная фраза
    prompt = "PyTorch is amazing because"

    # 3. Кодируем текст в тензоры (формат данных PyTorch)
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # 4. Генерируем продолжение текста
    output = model.generate(inputs, max_length=30, num_return_sequences=1,
                            no_repeat_ngram_size=2)

    # 5. Декодируем числа обратно в понятный текст
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)


if __name__ == "__main__":
    main()


# [transformers] The attention mask and the pad token id were not set. As a consequence, you may
# observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

# [transformers] Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

# [transformers] The attention mask is not set and cannot be inferred from input because pad token
# is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your
# input's `attention_mask` to obtain reliable results.
