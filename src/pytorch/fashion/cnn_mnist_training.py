"""
Вот полный рабочий код на PyTorch, который:

* загружает Fashion MNIST,
* визуализирует случайные товары,
* строит сверточную нейросеть (CNN) и обучает её
* классифицировать одежду по картинке.

Как устроен этот процесс классификации:

* Визуализация (show_samples): Извлекает случайные тензоры, убирает лишнюю размерность 
  с помощью .squeeze() и отрисовывает серую сетку с подписями через matplotlib.
* Сверточные слои (Conv2d): Сканируют картинку небольшими окнами (3x3), 
  выявляя границы, текстуры и очертания конкретного типа одежды.
* Пулинг (MaxPool2d): Сжимает картинку, оставляя самые важные пиксели, чтобы сеть 
  училась быстрее и не зависела от мелких сдвигов товара в кадре.
* Предсказание (argmax): Нейросеть выдает 10 чисел. Команда
  output.argmax(1) находит позицию самого большого числа, которая указывает на
  итоговый тип товара (например, «Кроссовки»).
"""

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================
train_data = datasets.FashionMNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", 
    train=False, 
    download=True, 
    transform=ToTensor()
)

# Словарь текстовых названий классов
classes = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# Создаем загрузчики (батч-размер 64)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ==========================================
# 2. ВИЗУАЛИЗАЦИЯ ИЗОБРАЖЕНИЙ
# ==========================================
def show_samples(dataset, num_samples=9):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, num_samples + 1):
        # Берем случайный индекс
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]

        figure.add_subplot(rows, cols, i)
        plt.title(classes[label])
        plt.axis("off")
        # Избавляемся от одиночного канала цвета [1, 28, 28] -> [28, 28] для matplotlib
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# Показываем картинки перед обучением
show_samples(train_data)

# ==========================================
# 3. АРХИТЕКТУРА НЕЙРОСЕТИ (CNN)
# ==========================================
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Вход: 1 канал цвета, выход: 16 каналов, фильтр 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # Уменьшает размер картинки в 2 раза

        # Вход: 16 каналов, выход: 32 канала
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Полносвязные слои (после двух пулингов размер 28x28 упал до 7x7)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 выходов на 10 классов одежды

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Размер становится 14x14
        x = self.pool(self.relu(self.conv2(x)))  # Размер становится 7x7
        x = torch.flatten(x, 1)  # Вытягиваем матрицу признаков в один вектор
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Модель запущена на устройстве: {device}")

# ==========================================
# 4. ОБУЧЕНИЕ НЕЙРОСЕТИ (1 эпоха для примера)
# ==========================================
model.train()
for batch, (X, y) in enumerate(train_loader):
    X, y = X.to(device), y.to(device)

    # Прямой проход (вычисление предсказания)
    pred = model(X)
    loss = criterion(pred, y)

    # Обратный проход (вычисление градиентов и обновление весов)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 200 == 0:
        print(f"Батч {batch}/{len(train_loader)} | Ошибка (Loss): {loss.item():.4f}")

# ==========================================
# 5. ПРОВЕРКА КЛАССИФИКАЦИИ ТОВАРА
# ==========================================
model.eval()
test_img, test_label = test_data[0]  # Берем самый первый товар из теста

with torch.no_grad():
    # Добавляем размерность батча [1, 1, 28, 28] и отправляем на устройство
    input_tensor = test_img.unsqueeze(0).to(device)
    output = model(input_tensor)
    # Находим индекс максимального значения (наиболее вероятный класс)
    predicted_idx = output.argmax(1).item()

# Визуализация результата предсказания
plt.imshow(test_img.squeeze(), cmap="gray")
plt.title(
    f"Реальный класс: {classes[test_label]}\nПредсказание: {classes[predicted_idx]}"
)
plt.axis("off")
plt.show()
