"""
Пример с использованием `pytorch` обучения с учителем (:term:`supervised learning`).

* `nn` - модуль для работы с нейросетями;
* `torch.tensor` - 

"""

import torch
import torch.nn as nn

# --- Данные ---
# Задача: предсказать y = 2x + 1
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # входные признаки
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # целевые значения

# --- Модель: один линейный нейрон y = w*x + b ---
model = nn.Linear(in_features=1, out_features=1)

# --- Функция потерь и оптимизатор ---
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# --- Обучение ---
for epoch in range(200):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.6f}")

# --- Результат ---
w = model.weight.item()
b = model.bias.item()
print(f"\nОбученные параметры: w={w:.4f}, b={b:.4f}")
print(f"Ожидалось:           w=2.0000, b=1.0000")

# Предсказание для нового значения x=5
x_new = torch.tensor([[5.0]])
print(f"\nПредсказание для x=5: {model(x_new).item():.4f}  (ожидается ~11.0)")

