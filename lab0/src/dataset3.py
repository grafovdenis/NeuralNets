# Third dataset

import matplotlib.pyplot as plt
import random as rand
import numpy as np
import sys

sys.path.append('../..')

from lib.draw import *


def load_data(train_size=3000, show=False):
    test_size = int(train_size * 0.2)

    x_train = np.empty(0)
    y_train = np.empty(0)

    x_test = np.empty(0)
    y_test = np.empty(0)

    # Zero class
    x0_train = np.empty(0)
    x0_test = np.empty(0)

    # First class
    x1_train = np.empty(0)
    x1_test = np.empty(0)

    for i in range(train_size + test_size):
        x = rand.uniform(0, 1)
        y = rand.uniform(0, 1)
        if i < train_size:
            x_train = np.append(x_train, (x, y))
        else:
            x_test = np.append(x_test, (x, y))

        # Text
        if draw_rectangle(x, y, 0.2, 0.2, 0.6, 0.6) | draw_triangle(x, y, 0.2, 0.6, 0.4, 0.8, 0.6, 0.6):
            if i < train_size:
                x0_train = np.append(x0_train, (x, y))
                y_train = np.append(y_train, 1)
            else:
                x0_test = np.append(x0_test, (x, y))
                y_test = np.append(y_test, 1)
        else:
            if i < train_size:
                x1_train = np.append(x1_train, (x, y))
                y_train = np.append(y_train, 0)
            else:
                x1_test = np.append(x1_test, (x, y))
                y_test = np.append(y_test, 0)

    # Reshaping
    x_train.shape = (train_size, 2)
    x_test.shape = (test_size, 2)

    x0_train.shape = (int(x0_train.size / 2), 2)
    x1_train.shape = (int(x1_train.size / 2), 2)

    x0_test.shape = (int(x0_test.size / 2), 2)
    x1_test.shape = (int(x1_test.size / 2), 2)

    # Plotting train
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("train data")
    plt.plot(x0_train.transpose()[0], x0_train.transpose()[1], '.')
    plt.plot(x1_train.transpose()[0], x1_train.transpose()[1], '.')

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    if show:
        plt.show()

    plt.close()

    # Plotting test
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("test data")
    plt.plot(x0_test.transpose()[0], x0_test.transpose()[1], '.')
    plt.plot(x1_test.transpose()[0], x1_test.transpose()[1], '.')

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    if show:
        plt.show()

    return (x_train, y_train), (x_test, y_test)
