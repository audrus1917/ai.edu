"""Пример обучения с учителем в PyTorch.

Используются:
- torch.tensor для хранения входных и целевых данных;
- torch.nn.Linear для линейной модели;
- MSELoss и SGD для обучения.

Модель должна восстановить зависимость y = 2x + 1 по нескольким точкам.
"""

import torch
import torch.nn as nn


RANDOM_SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 200
LOG_EVERY = 50


def main() -> None:
    """Задача: предсказать ``y = 2x + 1``."""

    torch.manual_seed(RANDOM_SEED)

    # Данные
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # входные признаки
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # целевые значения

    # Модель: один линейный нейрон y = w * x + b
    model = nn.Linear(in_features=1, out_features=1)

    # Функция потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Прямой проход: модель вычисляет предсказание по текущим параметрам.
    # Обратный проход: loss.backward() считает градиенты, optimizer.step() обновляет веса.
    for epoch in range(EPOCHS):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_EVERY == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.6f}")

    # --- Результат ---
    w = model.weight.item()
    b = model.bias.item()
    print(f"\nОбученные параметры: w={w:.4f}, b={b:.4f}")
    print("Ожидалось:           w=2.0000, b=1.0000")

    # Предсказание для нового значения x=5
    x_new = torch.tensor([[5.0]])
    print(f"\nПредсказание для x=5: {model(x_new).item():.4f}  (ожидается ~11.0)")


if __name__ == "__main__":
    main()
