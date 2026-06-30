"""
Пример кода на PyTorch, который загружает датасет Fashion MNIST, преобразует
изображения в тензоры и создает загрузчики данных (DataLoader) для обучения
нейросети.
"""

import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1. Загрузка обучающей выборки
training_data = datasets.FashionMNIST(
    root="data",         # Папка для сохранения данных
    train=True,          # Загрузить обучающий набор
    download=True,       # Скачать из интернета, если нет локально
    transform=ToTensor() # Преобразовать изображение PIL в тензор PyTorch (0-1)
)

# 2. Загрузка тестовой выборки
test_data = datasets.FashionMNIST(
    root="data",
    train=False,         # Загрузить тестовый набор
    download=True,
    transform=ToTensor()
)

# 3. Создание DataLoader для обхода данных батчами
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 4. Проверка структуры данных
for X, y in train_dataloader:
    print(f"Формат тензора изображений [B, C, H, W]: {X.shape}")
    print(f"Формат тензора меток классов: {y.shape} | Тип: {y.dtype}")
    break
