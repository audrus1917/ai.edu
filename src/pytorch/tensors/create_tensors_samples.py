#!/usr/bin/env python

"""
Примеры работы с тензорами:

* создание тензоров
* операции над тензорами

Типы данных в тензорах:

* torch.float16 - 16 бит, с плавающей точкой ( half precision, используется для экономии памяти и ускорения вычислений на GPU)
* torch.float32 - 32 бита,  с плавающей точкой
* torch.float64 - 64 бита, с плавающей точкой
* torch.int8 - 8 бит, целые числа (используется для экономии памяти и ускорения вычислений на GPU)
* torch.int16 - 16 бит, целые числа
* torch.int32 - 32 бита, целые числа
* torch.int64 - 64 бита, целые числа (обычно используется для индексов и счетчиков)
* torch.bool - булевы значения (True/False)
"""

import torch
import numpy as np


def main():
    """Примеры работы с тензорами."""

    create_tensors_samples()
    tensor_main_properties()


def create_tensors_samples():
    """Примеры создания тензоров."""

    # По умолчанию dtype = torch.float32
    tensor_simple = torch.Tensor(3, 5, 2)
    print(f"Простой тензор (пример 1): {tensor_simple}")

    tensor_simple = torch.empty(3, 5, 2)
    print(f"Простой тензор (пример 2): {tensor_simple}")

    # Другой `dtype`
    tensor_another_dtype = torch.empty(3, 5, 2, dtype=torch.int32)
    print(f"Тензор с другим dtype: {tensor_another_dtype}")

    t1 = torch.tensor([1]) # одно целочисленное значение 1
    print(f"Тензор t1: {t1}, dtype: {t1.dtype}")
    t2 = torch.tensor([1, 2.0]) # два вещественных значения 1.0 и 2.0
    print(f"Тензор t2: {t2}, dtype: {t2.dtype}")
    t3 = torch.tensor([[1, 2], [3, 4], [5, 6]]) # тензор 3x2 с целочисленным типом
    print(f"Тензор t3: {t3}, dtype: {t3.dtype}")
    t4 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    print(f"Тензор t4: {t4}, dtype: {t4.dtype}")

    # Создание тензора из NumPy массива
    numpy_array = np.array([[5, 6], [7, 8]])
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print("\nТензор из NumPy массива:")
    print(tensor_from_numpy)

    # Создание тензора с помощью функции arange
    tensor_arange = torch.arange(0, 10, 2)
    print("\nТензор с помощью arange:")
    print(tensor_arange)

    # Создание тензора с помощью функции linspace
    tensor_linspace = torch.linspace(0, 1, steps=5)
    print("\nТензор с помощью linspace:")
    print(tensor_linspace)

    # Создание тензора с помощью функции rand
    tensor_rand = torch.rand(2, 3)
    print("\nТензор с помощью rand:")
    print(tensor_rand)


def tensor_main_properties():
    """Основные свойства тензоров."""

    tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    print(f"Тензор: {tensor}")
    print(f"Размерность (shape): {tensor.shape}")
    print(f"Количество элементов (numel): {tensor.numel()}")
    print(f"Тип данных (dtype): {tensor.dtype}")
    print(f"Устройство (device): {tensor.device}")


if __name__ == "__main__":
    main()
