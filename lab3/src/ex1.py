# Task 1
# Compare 3 value of spread
# spread less than necessary,spread optimal and spread more than necessary
import sys

sys.path.append("../..")

import lab0.src.dataset5 as dataset5
from lib.RBF.RBFN import RBFN

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("Сравним 3 значения параметра spread для обучения RBFN на исходных данных номер 5")

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=15000, show=False)

    neu_num = 50
    titles = ["spread less than necessary", "spread optimal", "spread greater than necessary"]
    spreads = [0.01, 0.5, 20.0]

    for title, spread in zip(titles, spreads):
        model = RBFN(hidden_shape=neu_num, sigma=spread)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mae = (np.abs(y_test - y_pred)).mean()

        plt.plot(x_test, y_test, 'b.', label='real')
        plt.plot(x_test, y_pred, 'r.', label='fit')
        plt.legend(loc='upper left')
        plt.title(title + '\nRBFN\n neu_num = %.d\n spread = %.3f\n mae = %.4f' %
                  (neu_num, spread, mae))
        plt.show()

    # Dependence of mae from spread in the logarithmic scale
    print("\n\nПостроим график зависимости ошибки аппроксимации от параметра spread в логарифмическом масштабе")
    print("(Это надолго)")

    maes = np.empty(0)
    spreads = np.empty(0)

    for spread in np.arange(0.1, 10, step=0.5):
        model = RBFN(hidden_shape=neu_num, sigma=np.log(spread))
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        maes = np.append(maes, (np.abs(y_test - y_pred)).mean())
        spreads = np.append(spreads, np.log(spread))

    plt.plot(spreads, maes)
    plt.title('Dependence of mae from spread in the logarithmic scale')
    plt.xlabel("log(spread)")
    plt.ylabel("mae")
    plt.show()
