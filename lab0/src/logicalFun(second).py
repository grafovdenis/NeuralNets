# Second dataset

import numpy as np


# Displays squares of numbers between 0 and 31
def func(x1: int, x2: int, x3: int, x4: int, x5: int):
    return (not x1) & (not x2) & (not x3) & (not x4) & x5 \
           | (not x1) & (not x2) & x3 & (not x4) & (not x5) \
           | x1 & (not x2) & (not x3) & (not x4) & (not x5) \
           | x2 & (not x3) & (not x4) & x5 | x1 & x2 & (not x3) & (not x4) & x5


def load_data():
    x_train = np.empty(0)
    y_train = np.empty(0)
    for x1 in range(0, 2):
        for x2 in range(0, 2):
            for x3 in range(0, 2):
                for x4 in range(0, 2):
                    for x5 in range(0, 2):
                        x_train = np.append(x_train, np.array([x1, x2, x3, x4, x5]))
                        y_train = np.append(y_train, func(x1, x2, x3, x4, x5))

    return (x_train.reshape(32, 5), y_train)


(x_train, y_train) = load_data()

print("Table of truth:")
for i in range(0, 32):
    print(x_train[i], "=", y_train[i])
