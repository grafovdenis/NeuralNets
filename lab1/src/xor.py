import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import sys

from keras.utils import plot_model

sys.path.append("../..")
from lib.custom import hard_lim

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# creating model with weights initialization
model = Sequential()

model.add(Dense(2, input_dim=2, activation=hard_lim, name='First',
                weights=list([np.array([[1.0, -1.0],
                                        [1.0, -1.0]], dtype=float),
                              np.array([-0.5, 1.5], dtype=float)])))
model.add(Dense(1, activation=hard_lim, name='Second',
                weights=list([np.array([[1.0], [1.0]], dtype=float),
                              np.array([-1.5], dtype=float)])))

plot_model(model, to_file='../img/xor.png', show_shapes=True, show_layer_names=True)

# checking that model is working correctly

print("result:\n", np.array_equal(model.predict(x_train), y_train))
print(model.summary())
