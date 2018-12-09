# Fifth dataset

import numpy as np
import matplotlib.pyplot as plt


def function(t):
    return np.abs(np.sin(5 * t) * np.cos(20 * np.pi * t) * np.arctan(t - 1) - 0.5) * 0.8


def load_data(train_size=200, show=False, save=False):
    test_size = int(train_size * 0.2)
    h = 1 / float(train_size + test_size)

    x_train = np.empty(0)
    y_train = np.empty(0)
    x_test = np.empty(0)
    y_test = np.empty(0)

    for i, x in enumerate(np.arange(0.0, 1.0, h, dtype=float)):
        if i % 5 == 0:
            x_test = np.append(x_test, x)
            y_test = np.append(y_test, function(x))
        else:
            x_train = np.append(x_train, x)
            y_train = np.append(y_train, function(x))
    plt.plot(x_train, y_train, '.')
    plt.plot(x_test, y_test, '.')
    plt.legend(('train_data', 'test_data'), shadow=True)
    if save:
        plt.savefig('../plots/fifth.png')
    if show:
        plt.show()
    return (x_train, y_train), (x_test, y_test)
