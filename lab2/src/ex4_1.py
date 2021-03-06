# Task 4.1
# Классификация с помощью НС ПР, 1 слой
from __future__ import print_function
import sys

sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

import lib.gui_reporter as gr
from lib.custom import EarlyStoppingByLossVal
from lab0.src import dataset3

if __name__ == '__main__':
    np.random.seed(18)
    # 1,2 initializing
    train_size = 20000
    batch_size = 100
    epochs = 300
    verbose = 1
    neurons = 20
    lr = 0.005

    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    goal_loss = 0.0001

    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size, show=True)

    model = Sequential()

    model.add(Dense(neurons, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform',
                    activation='relu'))

    model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    # 3 setting stopper
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

    # 4 model fitting
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test))

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save=False, show=True)

    plt_x_zero = np.empty(0)
    plt_y_zero = np.empty(0)

    plt_x_one = np.empty(0)
    plt_y_one = np.empty(0)

    for i in x_test:
        if model.predict(np.array([i[0], i[1]]).reshape(1, 2)) >= 0.5:
            plt_x_zero = np.append(plt_x_zero, i[0])
            plt_y_zero = np.append(plt_y_zero, i[1])
        else:
            plt_x_one = np.append(plt_x_one, i[0])
            plt_y_one = np.append(plt_y_one, i[1])

    plt.plot(plt_x_zero, plt_y_zero, '.')
    plt.plot(plt_x_one, plt_y_one, '.')

    plt.title('classification\nlr = %.3f\nval_loss = %.4f\n neurons = %.d' % (
        lr, history.history["val_loss"][history.epoch.__len__() - 1], neurons))

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    plt.show()
    plt.close()
