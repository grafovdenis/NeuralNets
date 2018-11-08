# Решение - в лоб (рандомим, пока не будет по 8 элементов и проверяем на разделимость)

import numpy as np
import matplotlib.pyplot as plt


def isSeparable(matrix):
    if matrix.sum() == 8:
        # проверка по горизонтали
        if sum(matrix[0]) == sum(matrix[1]) or sum(matrix[2]) == sum(matrix[3]):
            return True
        # проверка по вертикали
        matrix = np.matrix.transpose(matrix)
        if sum(matrix[0]) == sum(matrix[1]) or sum(matrix[2]) == sum(matrix[3]):
            return True
    return False


matrix = np.round(np.random.rand(4, 4))

while not isSeparable(matrix):
    matrix = np.round(np.random.rand(4, 4))

print(matrix)

plt.imshow(matrix, cmap='gray')
plt.show()
