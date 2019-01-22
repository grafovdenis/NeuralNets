# Fourth dataset

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../..')

from lib.draw import *


def load_data(train_size=4000, show=False):
    test_size = int(train_size * 0.2)

    x_train = np.empty(0)
    y_train = np.empty(0)

    x_test = np.empty(0)
    y_test = np.empty(0)

    train_plt_points0 = np.empty(0)
    train_plt_points1 = np.empty(0)
    train_plt_points2 = np.empty(0)
    train_plt_points3 = np.empty(0)
    train_plt_points4 = np.empty(0)
    train_plt_points5 = np.empty(0)
    train_plt_points6 = np.empty(0)

    test_plt_points0 = np.empty(0)
    test_plt_points1 = np.empty(0)
    test_plt_points2 = np.empty(0)
    test_plt_points3 = np.empty(0)
    test_plt_points4 = np.empty(0)
    test_plt_points5 = np.empty(0)
    test_plt_points6 = np.empty(0)

    for i in range(0, train_size + test_size):
        x = np.random.random()
        y = np.random.random()

        # First class
        if draw_rectangle(x, y, .1, .1, .3, .3) | draw_triangle(x, y, .1, .3, .2, .4, .3, .3):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([1, 0, 0, 0, 0, 0, 0]))
                train_plt_points1 = np.append(train_plt_points1, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([1, 0, 0, 0, 0, 0, 0]))
                test_plt_points1 = np.append(test_plt_points1, (x, y))

        # Second class
        elif draw_rectangle(x, y, .4, .1, .6, .4) | draw_triangle(x, y, .4, .4, .6, .6, .6, .4):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 1, 0, 0, 0, 0, 0]))
                train_plt_points2 = np.append(train_plt_points2, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([0, 1, 0, 0, 0, 0, 0]))
                test_plt_points2 = np.append(test_plt_points2, (x, y))

        # Third class
        elif draw_rectangle(x, y, .7, .1, .9, .4) | draw_triangle(x, y, .7, .4, .7, .6, .9, .4):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 1, 0, 0, 0, 0]))
                train_plt_points3 = np.append(train_plt_points3, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([0, 0, 1, 0, 0, 0, 0]))
                test_plt_points3 = np.append(test_plt_points3, (x, y))

        # Fourth class
        elif draw_rectangle(x, y, .1, .8, .4, .9):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 1, 0, 0, 0]))
                train_plt_points4 = np.append(train_plt_points4, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([0, 0, 0, 1, 0, 0, 0]))
                test_plt_points4 = np.append(test_plt_points4, (x, y))

        # Fifth class
        elif draw_rectangle(x, y, .7, .7, .9, .9):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 1, 0, 0]))
                train_plt_points5 = np.append(train_plt_points5, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([0, 0, 0, 0, 1, 0, 0]))
                test_plt_points5 = np.append(test_plt_points5, (x, y))

        # Sixth class
        elif draw_rectangle(x, y, .1, .6, .2, .7):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 0, 1, 0]))
                train_plt_points6 = np.append(train_plt_points6, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([0, 0, 0, 0, 0, 1, 0]))
                test_plt_points6 = np.append(test_plt_points6, (x, y))

        # Seventh class
        else:
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 0, 0, 1]))
                train_plt_points0 = np.append(train_plt_points0, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([0, 0, 0, 0, 0, 0, 1]))
                test_plt_points0 = np.append(test_plt_points0, (x, y))

    # reshaping
    x_train.shape = (int(x_train.size / 2), 2)
    y_train.shape = (int(y_train.size / 7), 7)
    train_plt_points0.shape = (int(train_plt_points0.size / 2), 2)
    train_plt_points1.shape = (int(train_plt_points1.size / 2), 2)
    train_plt_points2.shape = (int(train_plt_points2.size / 2), 2)
    train_plt_points3.shape = (int(train_plt_points3.size / 2), 2)
    train_plt_points4.shape = (int(train_plt_points4.size / 2), 2)
    train_plt_points5.shape = (int(train_plt_points5.size / 2), 2)
    train_plt_points6.shape = (int(train_plt_points6.size / 2), 2)

    x_test.shape = (int(x_test.size / 2), 2)
    y_test.shape = (int(y_test.size / 7), 7)
    test_plt_points0.shape = (int(test_plt_points0.size / 2), 2)
    test_plt_points1.shape = (int(test_plt_points1.size / 2), 2)
    test_plt_points2.shape = (int(test_plt_points2.size / 2), 2)
    test_plt_points3.shape = (int(test_plt_points3.size / 2), 2)
    test_plt_points4.shape = (int(test_plt_points4.size / 2), 2)
    test_plt_points5.shape = (int(test_plt_points5.size / 2), 2)
    test_plt_points6.shape = (int(test_plt_points6.size / 2), 2)

    # plotting
    plt.xlim(0, 1.3)
    plt.ylim(0, 1)
    plt.title("train data")
    plt.plot(train_plt_points0.transpose()[0], train_plt_points0.transpose()[1], '.')
    plt.plot(train_plt_points1.transpose()[0], train_plt_points1.transpose()[1], '.')
    plt.plot(train_plt_points2.transpose()[0], train_plt_points2.transpose()[1], '.')
    plt.plot(train_plt_points3.transpose()[0], train_plt_points3.transpose()[1], '.')
    plt.plot(train_plt_points4.transpose()[0], train_plt_points4.transpose()[1], '.')
    plt.plot(train_plt_points5.transpose()[0], train_plt_points5.transpose()[1], '.')
    plt.plot(train_plt_points6.transpose()[0], train_plt_points6.transpose()[1], '.')

    plt.legend(('1 class', '2 class', '3 class', '4 class', '5 class', '6 class',
                '7 class'), loc='upper right', shadow=True)

    if show:
        plt.show()

    plt.close()

    # plotting
    plt.xlim(0, 1.3)
    plt.ylim(0, 1)
    plt.title("test data")
    plt.plot(test_plt_points0.transpose()[0], test_plt_points0.transpose()[1], '.')
    plt.plot(test_plt_points1.transpose()[0], test_plt_points1.transpose()[1], '.')
    plt.plot(test_plt_points2.transpose()[0], test_plt_points2.transpose()[1], '.')
    plt.plot(test_plt_points3.transpose()[0], test_plt_points3.transpose()[1], '.')
    plt.plot(test_plt_points4.transpose()[0], test_plt_points4.transpose()[1], '.')
    plt.plot(test_plt_points5.transpose()[0], test_plt_points5.transpose()[1], '.')
    plt.plot(test_plt_points6.transpose()[0], test_plt_points6.transpose()[1], '.')

    plt.legend(('1 class', '2 class', '3 class', '4 class', '5 class', '6 class',
                '7 class'), loc='upper right', shadow=True)

    if show:
        plt.show()
    plt.close()

    return (x_train, y_train), (x_test, y_test)


def load_data_neupy(train_size=4000, show=False):
    test_size = int(train_size * 0.2)

    x_train = np.empty(0)
    y_train = np.empty(0)

    x_test = np.empty(0)
    y_test = np.empty(0)

    train_plt_points0 = np.empty(0)
    train_plt_points1 = np.empty(0)
    train_plt_points2 = np.empty(0)
    train_plt_points3 = np.empty(0)
    train_plt_points4 = np.empty(0)
    train_plt_points5 = np.empty(0)
    train_plt_points6 = np.empty(0)

    test_plt_points0 = np.empty(0)
    test_plt_points1 = np.empty(0)
    test_plt_points2 = np.empty(0)
    test_plt_points3 = np.empty(0)
    test_plt_points4 = np.empty(0)
    test_plt_points5 = np.empty(0)
    test_plt_points6 = np.empty(0)

    for i in range(0, train_size + test_size):
        x = np.random.random()
        y = np.random.random()

        # First class
        if draw_rectangle(x, y, .1, .1, .3, .3) | draw_triangle(x, y, .1, .3, .2, .4, .3, .3):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0.1)
                train_plt_points1 = np.append(train_plt_points1, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0.1)
                test_plt_points1 = np.append(test_plt_points1, (x, y))

        # Second class
        elif draw_rectangle(x, y, .4, .1, .6, .4) | draw_triangle(x, y, .4, .4, .6, .6, .6, .4):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0.2)
                train_plt_points2 = np.append(train_plt_points2, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0.2)
                test_plt_points2 = np.append(test_plt_points2, (x, y))

        # Third class
        elif draw_rectangle(x, y, .7, .1, .9, .4) | draw_triangle(x, y, .7, .4, .7, .6, .9, .4):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0.3)
                train_plt_points3 = np.append(train_plt_points3, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0.3)
                test_plt_points3 = np.append(test_plt_points3, (x, y))

        # Fourth class
        elif draw_rectangle(x, y, .1, .8, .4, .9):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0.4)
                train_plt_points4 = np.append(train_plt_points4, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0.4)
                test_plt_points4 = np.append(test_plt_points4, (x, y))

        # Fifth class
        elif draw_rectangle(x, y, .7, .7, .9, .9):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0.5)
                train_plt_points5 = np.append(train_plt_points5, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0.5)
                test_plt_points5 = np.append(test_plt_points5, (x, y))

        # Sixth class
        elif draw_rectangle(x, y, .1, .6, .2, .7):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0.6)
                train_plt_points6 = np.append(train_plt_points6, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0.6)
                test_plt_points6 = np.append(test_plt_points6, (x, y))

        # Zeros class
        else:

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0.0)
                train_plt_points0 = np.append(train_plt_points0, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0.0)
                test_plt_points0 = np.append(test_plt_points0, (x, y))

    # reshaping
    x_train.shape = (int(x_train.size / 2), 2)
    train_plt_points0.shape = (int(train_plt_points0.size / 2), 2)
    train_plt_points1.shape = (int(train_plt_points1.size / 2), 2)
    train_plt_points2.shape = (int(train_plt_points2.size / 2), 2)
    train_plt_points3.shape = (int(train_plt_points3.size / 2), 2)
    train_plt_points4.shape = (int(train_plt_points4.size / 2), 2)
    train_plt_points5.shape = (int(train_plt_points5.size / 2), 2)
    train_plt_points6.shape = (int(train_plt_points6.size / 2), 2)

    x_test.shape = (int(x_test.size / 2), 2)
    test_plt_points0.shape = (int(test_plt_points0.size / 2), 2)
    test_plt_points1.shape = (int(test_plt_points1.size / 2), 2)
    test_plt_points2.shape = (int(test_plt_points2.size / 2), 2)
    test_plt_points3.shape = (int(test_plt_points3.size / 2), 2)
    test_plt_points4.shape = (int(test_plt_points4.size / 2), 2)
    test_plt_points5.shape = (int(test_plt_points5.size / 2), 2)
    test_plt_points6.shape = (int(test_plt_points6.size / 2), 2)

    # plotting
    plt.xlim(0, 1.5)
    plt.ylim(0, 1)
    plt.title("train data")
    plt.plot(train_plt_points0.transpose()[0], train_plt_points0.transpose()[1], '.')
    plt.plot(train_plt_points1.transpose()[0], train_plt_points1.transpose()[1], '.')
    plt.plot(train_plt_points2.transpose()[0], train_plt_points2.transpose()[1], '.')
    plt.plot(train_plt_points3.transpose()[0], train_plt_points3.transpose()[1], '.')
    plt.plot(train_plt_points4.transpose()[0], train_plt_points4.transpose()[1], '.')
    plt.plot(train_plt_points5.transpose()[0], train_plt_points5.transpose()[1], '.')
    plt.plot(train_plt_points6.transpose()[0], train_plt_points6.transpose()[1], '.')

    plt.legend(('0.0 class', '0.1 class', '0.2 class', '0.3 class', '0.4 class', '0.5 class',
                '0.6 class'), loc='upper right', shadow=True)

    if show:
        plt.show()

    plt.close()

    # plotting
    plt.xlim(0, 1.5)
    plt.ylim(0, 1)
    plt.title("test data")
    plt.plot(test_plt_points0.transpose()[0], test_plt_points0.transpose()[1], '.')
    plt.plot(test_plt_points1.transpose()[0], test_plt_points1.transpose()[1], '.')
    plt.plot(test_plt_points2.transpose()[0], test_plt_points2.transpose()[1], '.')
    plt.plot(test_plt_points3.transpose()[0], test_plt_points3.transpose()[1], '.')
    plt.plot(test_plt_points4.transpose()[0], test_plt_points4.transpose()[1], '.')
    plt.plot(test_plt_points5.transpose()[0], test_plt_points5.transpose()[1], '.')
    plt.plot(test_plt_points6.transpose()[0], test_plt_points6.transpose()[1], '.')

    plt.legend(('0.0 class', '0.1 class', '0.2 class', '0.3 class', '0.4 class', '0.5 class',
                '0.6 class'), loc='upper right', shadow=True)
    if show:
        plt.show()
    plt.close()

    return (x_train, y_train), (x_test, y_test)
