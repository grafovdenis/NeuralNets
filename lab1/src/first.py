import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import sys

sys.path.append("../..")
from lib.custom import hard_lim
from lab0.src import dataset1

y_train = np.array([[1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1]])

# load data from dataset1
(x_train, y_train) = dataset1.load_data(y_train=y_train, show=True)


# creating model with weights initialization
model = Sequential()
model.add(Dense(4, input_dim=x_train.shape[1], activation=hard_lim, name='First',
                weights=list([np.array([[0.0, 1.0, 0.0, -1.0],
                                        [1.0, 0.0, -1.0, 0.0]], dtype=float),
                              np.array([-0.5, -0.5, 0.5, 0.5], dtype=float)])))
model.add(Dense(2, activation=hard_lim, name='Second',
                weights=list([np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=float),
                              np.array([-1.5, -1.5], dtype=float)])))
model.add(Dense(1, activation=hard_lim, name='Third',
                weights=list([np.array([[1.0], [1.0]], dtype=float),
                              np.array([-0.5], dtype=float)])))

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# checking that model is working correctly
compareResult = np.array_equal(model.predict(x_train).reshape(4, 4).reshape(y_train.shape), y_train)

print("y_train == model.predict(x_train):", compareResult)
print("result:\n", model.predict(x_train).reshape(4, 4))
print(model.summary())
