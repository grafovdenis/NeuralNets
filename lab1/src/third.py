import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import sys

sys.path.append("../..")
from lib.custom import hard_lim
from lab0.src import dataset3

train_size = 4000
# load data from dataset1
(x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size, show=True)

# creating model with weights initialization
model = Sequential()

model.add(Dense(12, input_dim=x_train.shape[1], activation=hard_lim, name='First',
                weights=list([np.array([[0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                                        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
                              np.array([-0.6, -0.2, -0.6, -0.2, -1.2, -0.4, -0.45, -0.35, -0.55, -0.45, -0.35, -0.25],
                                       dtype=float)])))

model.add(Dense(2, activation=hard_lim, name='Second',
                weights=list([np.array([[-1.0, 1.0],  # y1
                                        [1.0, 0.0],  # y2
                                        [-1.0, 0.0],  # y3
                                        [1.0, 0.0],  # y4
                                        [0.0, -1.0],  # y5
                                        [0.0, -1.0],  # y6
                                        [1.0, 0.0],  # y7
                                        [-1.0, 0.0],  # y8
                                        [1.0, 0.0],  # y9
                                        [-1.0, 0.0],  # y10
                                        [1.0, 0.0],  # y12
                                        [-1.0, 0.0]],  # y12
                                       dtype=float),
                              np.array([-1.5, -0.5], dtype=float)])))

model.add(Dense(1, activation=hard_lim, name='Third',
                weights=list([np.array([[1.0],
                                        [1.0]], dtype=float),
                              np.array([-0.5], dtype=float)])))

# checking that model is working correctly
right = 0
for i, pr in enumerate(model.predict(x_train)):
    if y_train[i] == pr:
        right += 1

print("Accuracy = %.f%%" % (right / float(train_size) * 100))
