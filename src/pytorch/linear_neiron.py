"""Пример использования линейного нейрона.

Используются:
- numpy для работы с массивами;
- собственная реализация линейного нейрона.

Модель должна восстановить зависимость y = 2x + 1 по нескольким точкам.
"""

import numpy as np


class LinearNeuron:
    """Линейный нейрон, который вычисляет взвешенную сумму входов и добавляет смещение (bias)."""

    def __init__(self, n_inputs):
        # Инициализация весов и смещения случайными значениями
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def forward(self, inputs):
        # Взвешенная сумма: (inputs * weights) + bias
        return np.dot(inputs, self.weights) + self.bias


if __name__ == "__main__":
    # Пример использования:
    neuron = LinearNeuron(n_inputs=3)
    sample_input = np.array([1.5, 2.0, 0.5])

    sample_input = np.array([1.5, 2.0, 2.5])
    
    prediction = neuron.forward(sample_input)

    print(f"Предсказание: {prediction}")

    # Для обучения модели можно использовать градиентный спуск или другой оптимизационный алгоритм,
    # но это выходит за рамки данного примера.
