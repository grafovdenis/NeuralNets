# First dataset
# Solution - head-on (random, until there are 8 elements each and we check for separability)

import matplotlib.pyplot as plt
import numpy as np


def is_separable(matrix):
    if matrix.sum() == 8:
        # horizontal check
        if sum(matrix[0]) == sum(matrix[1]) or sum(matrix[2]) == sum(matrix[3]):
            return True
        # vertical check
        matrix = np.matrix.transpose(matrix)
        if sum(matrix[0]) == sum(matrix[1]) or sum(matrix[2]) == sum(matrix[3]):
            return True
    return False


y_train = np.round(np.random.rand(4, 4))

while not is_separable(y_train):
    y_train = np.round(np.random.rand(4, 4))

print("classes:")
print(y_train)


def load_data(y_train, show=False, save=False):
    if y_train.shape != (4, 4):
        raise TypeError("Shape can be only (4,4)")

    train_size = int(y_train.shape[0] * y_train.shape[1])
    x_train = np.zeros(train_size * 2)

    first_class_x = np.zeros(8)
    first_class_y = np.zeros(8)
    second_class_x = np.zeros(8)
    second_class_y = np.zeros(8)

    i = 0
    one_i = 0
    zero_i = 0
    for y in np.arange(0, y_train.shape[0], 1, dtype=int):
        for x in np.arange(0, y_train.shape[1], 1, dtype=int):
            if y_train[y_train.shape[0] - y - 1][x] == 1:
                second_class_x[one_i] = x / 4.0 + 0.125
                second_class_y[one_i] = y / 4.0 + 0.125
                one_i += 1
            elif y_train[y_train.shape[0] - y - 1][x] == 0:
                first_class_x[zero_i] = x / 4.0 + 0.125
                first_class_y[zero_i] = y / 4.0 + 0.125
                zero_i += 1

            x_train[i] = y / 4.0 + 0.125
            x_train[i + 1] = x / 4.0 + 0.125
            i += 2

    plt.plot(first_class_x, first_class_y, '.')
    plt.plot(second_class_x, second_class_y, '.')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1)

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    if save:
        plt.savefig('../plots/first.png')
    if show:
        plt.show()
    plt.close()

    return (x_train[::-1].reshape(train_size, 2), y_train.reshape(train_size, 1))


# ---------------------------

(x_train, y_train) = load_data(y_train, show=True, save=False)
