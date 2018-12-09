import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import sys

sys.path.append("../..")
from lib.custom import hard_lim
from lab0.src import dataset2

# load data from dataset2
(x_train, y_train) = dataset2.load_data()

# creating model with weights initialization
model = Sequential()

DNF = np.array([[-1, -1, -1, -1, 1],
                [-1, -1, 1, -1, -1],
                [1, -1, -1, -1, -1],
                [0, 1, -1, -1, 1]])

model.add(Dense(4, input_dim=x_train.shape[1], activation=hard_lim, name='First',
                weights=list([np.transpose(DNF),
                              np.array([-0.5, -0.5, -0.5, -1.5], dtype=float)])))
model.add(Dense(1, activation=hard_lim, name='Second',
                weights=list(
                    [np.array([[1.0], [1.0], [1.0], [1.0]], dtype=float), np.array([-0.5], dtype=float)])))

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print("СДНФ")
print(dataset2.func.__doc__)
# checking that model is working correctly
compareResult = np.array_equal(model.predict(x_train).reshape(y_train.shape), y_train)
print("\ny_train == model.predict(x_train):", compareResult)
