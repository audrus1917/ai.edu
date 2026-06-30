#!/usr/bin/env python

"""Примеры автозаполнения и изменения формы тензоров."""

import torch
import numpy as np


def main():
    """Примеры автозаполнения и изменения формы тензоров."""

    tensor_zeros = torch.zeros(3, 4)
    print(f"Тензор, заполненный нулями: {tensor_zeros}")

    tensor_ones = torch.ones(2, 5)
    print(f"Тензор, заполненный единицами: {tensor_ones}")

    tensor_rand = torch.rand(3, 3)
    print(f"Тензор, заполненный случайными числами: {tensor_rand}")

    t1 = torch.eye(3) # тензор 3x3 с единицами по главной диагонали
    print(f"Тензор t1: {t1}, dtype: {t1.dtype}")
    t2 = torch.eye(3, 2) # тензор 3x2 с единицами по главной диагонали
    print(f"Тензор t2: {t2}, dtype: {t2.dtype}")
    t3 = torch.eye(3, 2, dtype=torch.int8) # тензор 3x2 с единицами по главной диагонали и типом данных int8
    print(f"Тензор t3: {t3}, dtype: {t3.dtype}")


if __name__ == "__main__":
    main()
