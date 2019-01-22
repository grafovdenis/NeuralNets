# Task 1.3 GRNN exploring
from __future__ import print_function
from neupy import algorithms
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../..")
import lab0.src.dataset5 as dataset5

(x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=6000, show=False)

maes = np.empty(0)
train_sizes = [6000]

print("\n\nВ данной реализации функция которую реализует GRNN может "
      "быть функционально очень похожа на исходную функцию\n")
print("однако mae при этом может быть довольно высокой")
print("Приведем пример поиска оптимального значения параметра speread для размера выборки = 6000")

goal_loss = 0.01
mae = 0.2
spread = 0.1
while mae > goal_loss:
    print("\n\n--------------spread = %.10f-------------------\n\n" % spread)
    for train_size in train_sizes:
        test_size = int(train_size * 0.2)

        nw = algorithms.GRNN(std=spread, verbose=False)

        nw.train(x_train[0:train_size], y_train[0:train_size])

        y_pred = nw.predict(x_test[0:test_size])

        mae = (np.abs(y_test[0:test_size] - y_pred)).mean()

        plt.plot(x_test[0:test_size], y_test[0:test_size], 'b.', label='real')
        plt.plot(x_test[0:test_size], y_pred, 'r.', label='fit')
        plt.legend(loc='upper right')
        plt.title('GRNN aprox with neupy\n train_size = %.d\n spread = %.10f\n mae = %.4f' % (train_size, spread, mae))
        plt.show()
    print(spread)
    print(mae)
    spread *= 0.1

print("\n\nОптимальный параметр spread для выборки размером 6000:")
print("spread = ", spread)
