# Task 1.3 GRNN exploring
from __future__ import print_function
from neupy import algorithms
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../..")
import lab0.src.dataset5 as dataset5

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=15000, show=False)

    train_sizes = [6000, 3000, 1000]
    spreads = [0.1, 0.01, 0.001, 0.0001]
    print("\n\nОпределим оптимальные значения параметра spread для выборок размером: \n", train_sizes[0],
          train_sizes[1], train_sizes[2])

    for train_size in train_sizes:
        goal_loss = 0.15
        mae = 0.2
        spread = 0.1
        print("\n\n--------------train_size = %.d-------------------\n\n" % train_size)
        i = 0
        while mae > goal_loss:
            test_size = int(train_size * 0.2)

            nw = algorithms.GRNN(std=spread, verbose=False)

            nw.train(x_train[0:train_size], y_train[0:train_size])

            y_pred = nw.predict(x_test[0:test_size])

            mae = (np.abs(y_test[0:test_size] - y_pred)).mean()
            spread *= 0.1

        plt.plot(x_test[0:test_size], y_test[0:test_size], 'b.', label='real')
        plt.plot(x_test[0:test_size], y_pred, 'r.', label='fit')
        plt.legend(loc='upper right')
        plt.title('GRNN aprox with neupy\n train_size = %.d\n spread = %.10f\n mae = %.4f' % (train_size, spread, mae))
        plt.show()

        print("\n\nОптимальный параметр spread для выборки размером %.d:" % train_size)
        print("spread = ", spread)
