import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import sys

sys.path.append("../..")
from lib.custom import hard_lim
from lab0.src import dataset4

train_size = 16000

(x_train, y_train), (x_test, y_test) = dataset4.load_data(train_size=train_size, show=True)

model = Sequential()

x1 = [.0, .0, .0, .0, .0, .0, .0, 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., -1.]
x2 = [1., 1., 1., 1., 1., 1., 1., .0, .0, .0, .0, .0, .0, .0, 1., 1., 1., 1.]
b = [-.9, -.8, -.7, -.6, -.4, -.3, -.1, -.9, -.7, -.6, -.4, -.3, -.2, -.1, -1.3, -0.6, 0.0, -0.2]

model.add(Dense(18, input_dim=x_train.shape[1], activation=hard_lim, name='First',
                weights=list([np.array([x1, x2], dtype=float),
                              np.array(b, dtype=float)])))

model.add(Dense(9, activation=hard_lim, name='Second',
                weights=list([np.array([[.0, .0, .0, .0, .0, .0, -1., -1., .0],  # y1
                                        [.0, .0, .0, .0, .0, .0, 1., .0, .0],  # y2
                                        [.0, .0, .0, .0, .0, .0, .0, 1., -1.],  # y3
                                        [.0, .0, .0, .0, .0, .0, .0, .0, 1.],  # y4
                                        [.0, .0, -1., 1., -1., 1., .0, .0, .0],  # y5
                                        [-1., 1., .0, .0, .0, .0, .0, .0, .0],  # y6
                                        [1., .0, 1., .0, 1., .0, .0, .0, .0],  # y7
                                        [.0, .0, .0, .0, -1., .0, .0, -1., .0],  # y8
                                        [.0, .0, .0, .0, 0., 1., .0, 1., .0],  # y9
                                        [.0, .0, -1., -1., .0, .0, .0, .0, .0],  # y10
                                        [.0, .0, 1., .0, .0, .0, -1., .0, .0],  # y11
                                        [-1., .0, .0, .0, .0, .0, .0, .0, .0],  # y12
                                        [.0, .0, .0, .0, .0, .0, .0, .0, -1.],  # y13
                                        [1., .0, .0, .0, .0, .0, 1., .0, 1.],  # y14
                                        [.0, .0, .0, .0, .0, -1., .0, .0, .0],  # y15
                                        [.0, -1., .0, .0, .0, .0, .0, .0, .0],  # y16
                                        [.0, .0, .0, -1., .0, .0, .0, .0, .0],  # y17
                                        [.0, -1., .0, .0, .0, .0, .0, .0, .0]], dtype=float),  # y18
                              np.array([-1.5, -0.5, -1.5, -0.5, -1.5, -1.5, -1.5, -1.5, -1.5], dtype=float)])))

model.add(Dense(6, activation=hard_lim, name='Third',
                weights=list([np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float),
                              np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], dtype=float)])))

i = 0
right = 0
for pr in model.predict(x_train):
    if np.array_equal(pr, y_train[i]):
        right += 1
    i += 1
print("Accuracy = %.f%%" % (right / float(train_size) * 100))
