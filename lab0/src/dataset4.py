# Fourth dataset

import matplotlib.pyplot as plt
import math
import numpy as np


def draw_rectangle(x, y, x1, y1, x2, y2):
    return (x1 < x < x2) & (y1 < y < y2)


def draw_circle(x, y, x1, y1, r):
    return not (x1 - x) ** 2 + (y1 - y) ** 2 < r ** 2


def draw_triangle(x, y, x1, y1, x2, y2, x3, y3):
    sign1 = (x1 - x) * (y2 - y1) - (x2 - x1) * (y1 - y)
    sign2 = (x2 - x) * (y3 - y2) - (x3 - x2) * (y2 - y)
    sign3 = (x3 - x) * (y1 - y3) - (x1 - x3) * (y3 - y)

    # Нормализация
    try:
        sign1 /= math.fabs(sign1)
        sign2 /= math.fabs(sign2)
        sign3 /= math.fabs(sign3)
    except ZeroDivisionError:
        return True

    return not int(sign1) == int(sign2) == int(sign3)


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
    train_plt_points7 = np.empty(0)

    test_plt_points0 = np.empty(0)
    test_plt_points1 = np.empty(0)
    test_plt_points2 = np.empty(0)
    test_plt_points3 = np.empty(0)
    test_plt_points4 = np.empty(0)
    test_plt_points5 = np.empty(0)
    test_plt_points6 = np.empty(0)
    test_plt_points7 = np.empty(0)

    for i in range(0, train_size + test_size):
        x = np.random.random()
        y = np.random.random()

        # First class
        if draw_rectangle(x, y, 0.05, 0.6, 0.1, 0.9) | draw_rectangle(x, y, 0.175, 0.6, 0.2, 0.9) \
                | draw_rectangle(x, y, 0.1, 0.725, 0.2, 0.775) | draw_rectangle(x, y, 0.1, 0.85, 0.2, 0.9) \
                | draw_rectangle(x, y, 0.1, 0.6, 0.2, 0.65):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([1, 0, 0, 0, 0, 0]))
                train_plt_points1 = np.append(train_plt_points1, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_train = np.append(y_train, np.array([1, 0, 0, 0, 0, 0]))
                test_plt_points1 = np.append(test_plt_points1, (x, y))

        # Second class
        elif draw_rectangle(x, y, 0.25, 0.6, 0.3, 0.9) \
                | draw_rectangle(x, y, 0.3, 0.725, 0.4, 0.775) \
                | draw_rectangle(x, y, 0.3, 0.85, 0.4, 0.9) \
                | draw_rectangle(x, y, 0.3, 0.6, 0.4, 0.65):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 1, 0, 0, 0, 0]))
                train_plt_points2 = np.append(train_plt_points2, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_train = np.append(y_train, np.array([0, 1, 0, 0, 0, 0]))
                test_plt_points2 = np.append(test_plt_points2, (x, y))
        # Third class
        elif draw_rectangle(x, y, 0.05, 0.1, 0.1, 0.4) \
                | draw_rectangle(x, y, 0.1, 0.225, 0.2, 0.275) \
                | draw_rectangle(x, y, 0.1, 0.35, 0.2, 0.4):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 1, 0, 0, 0]))
                train_plt_points3 = np.append(train_plt_points3, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 1, 0, 0, 0]))
                test_plt_points3 = np.append(test_plt_points3, (x, y))

        # Fourth class
        elif draw_rectangle(x, y, 0.25, 0.1, 0.3, 0.4) \
                | draw_rectangle(x, y, 0.375, 0.275, 0.4, 0.375) \
                | draw_rectangle(x, y, 0.375, 0.1, 0.4, 0.275) \
                | draw_rectangle(x, y, 0.3, 0.225, 0.4, 0.275) \
                | draw_rectangle(x, y, 0.3, 0.35, 0.4, 0.4):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 1, 0, 0]))
                train_plt_points4 = np.append(train_plt_points4, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 1, 0, 0]))
                test_plt_points4 = np.append(test_plt_points4, (x, y))

        # Fifth class
        elif draw_rectangle(x, y, 0.5, 0.1, 0.55, 0.4) \
                | draw_rectangle(x, y, 0.5, 0.225, 0.625, 0.275) \
                | draw_rectangle(x, y, 0.5, 0.35, 0.625, 0.4) \
                | draw_rectangle(x, y, 0.5, 0.1, 0.625, 0.15):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 1, 0]))
                train_plt_points5 = np.append(train_plt_points5, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 1, 0]))
                test_plt_points5 = np.append(test_plt_points5, (x, y))

        # Sixth class
        elif draw_rectangle(x, y, 0.75, 0.1, 0.8, 0.4) \
                | draw_rectangle(x, y, 0.75, 0.225, 0.875, 0.275) \
                | draw_rectangle(x, y, 0.75, 0.35, 0.875, 0.4) \
                | draw_rectangle(x, y, 0.75, 0.1, 0.875, 0.15):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 0, 1]))
                train_plt_points6 = np.append(train_plt_points6, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 0, 1]))
                test_plt_points6 = np.append(test_plt_points6, (x, y))

        # Seventh class
        elif draw_circle(x, y, 0.75, 0.75, 0.15) | (not draw_triangle(x, y, 0.65, 0.8, 0.8, 0.8, 0.8, 0.65)):
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 0, 1]))
                train_plt_points7 = np.append(train_plt_points7, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 0, 1]))
                test_plt_points7 = np.append(test_plt_points7, (x, y))

        # Eighth class
        else:
            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, np.array([0, 0, 0, 0, 0, 0]))
                train_plt_points0 = np.append(train_plt_points0, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, np.array([0, 0, 0, 0, 0, 0]))
                test_plt_points0 = np.append(test_plt_points0, (x, y))

    # reshaping
    x_train.shape = (int(x_train.size / 2), 2)
    y_train.shape = (int(y_train.size / 6), 6)
    train_plt_points0.shape = (int(train_plt_points0.size / 2), 2)
    train_plt_points1.shape = (int(train_plt_points1.size / 2), 2)
    train_plt_points2.shape = (int(train_plt_points2.size / 2), 2)
    train_plt_points3.shape = (int(train_plt_points3.size / 2), 2)
    train_plt_points4.shape = (int(train_plt_points4.size / 2), 2)
    train_plt_points5.shape = (int(train_plt_points5.size / 2), 2)
    train_plt_points6.shape = (int(train_plt_points6.size / 2), 2)
    train_plt_points7.shape = (int(train_plt_points7.size / 2), 2)

    x_test.shape = (int(x_test.size / 2), 2)
    y_test.shape = (int(y_test.size / 6), 6)
    test_plt_points0.shape = (int(test_plt_points0.size / 2), 2)
    test_plt_points1.shape = (int(test_plt_points1.size / 2), 2)
    test_plt_points2.shape = (int(test_plt_points2.size / 2), 2)
    test_plt_points3.shape = (int(test_plt_points3.size / 2), 2)
    test_plt_points4.shape = (int(test_plt_points4.size / 2), 2)
    test_plt_points5.shape = (int(test_plt_points5.size / 2), 2)
    test_plt_points6.shape = (int(test_plt_points6.size / 2), 2)
    test_plt_points7.shape = (int(test_plt_points7.size / 2), 2)

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
    plt.plot(train_plt_points7.transpose()[0], train_plt_points7.transpose()[1], '.')

    plt.legend(('1 class', '2 class', '3 class', '4 class', '5 class', '6 class',
                '7 class', '0 class'), loc='upper right', shadow=True)


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
    plt.plot(test_plt_points7.transpose()[0], test_plt_points7.transpose()[1], '.')

    plt.legend(('1 class', '2 class', '3 class', '4 class', '5 class', '6 class',
                '7 class', '0 class'), loc='upper right', shadow=True)

    if show:
        plt.show()

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data(train_size=16000, show=True)
