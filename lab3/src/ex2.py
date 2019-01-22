# Task 2, part 1
import matplotlib.pyplot as plt
import numpy as np
from neupy import algorithms

import lab0.src.dataset3 as dataset3

print("\n\nСравним и визуализируем случаи, когда spread больше, меньше и равен оптимальному значению для")
print("случая деления плоскости 2 класса\nВсе графики подписаны\n")

(x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=12000, show=False)
titles = ["\n\nspread greater than necessary", "\n\nspread optimal", "\n\nspread less than necessary"]
spreads = [0.1, 0.001, 0.0001]

for spread, title in zip(spreads, titles):
    pnn = algorithms.PNN(std=spread, verbose=False)

    pnn.train(x_train, y_train)

    y_predicted = pnn.predict(x_test)

    mae = (np.abs(y_test - y_predicted)).mean()

    plt_x_zero = np.empty(0)
    plt_y_zero = np.empty(0)

    plt_x_one = np.empty(0)
    plt_y_one = np.empty(0)

    acc = 0.0
    i = 0
    for coord in x_test:
        if y_predicted[i] < 0.5:
            plt_x_zero = np.append(plt_x_zero, coord[0])
            plt_y_zero = np.append(plt_y_zero, coord[1])
        elif y_predicted[i] >= 0.5:
            plt_x_one = np.append(plt_x_one, coord[0])
            plt_y_one = np.append(plt_y_one, coord[1])
        i += 1

    plt.plot(plt_x_zero, plt_y_zero, '.')
    plt.plot(plt_x_one, plt_y_one, '.')

    plt.title(title + '\n2 class classification\nstd = %.4f\nmae =%.4f' % (spread, mae))

    plt.xlim(0, 1.3)
    plt.ylim(0, 1)

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    plt.show()
    plt.close()
