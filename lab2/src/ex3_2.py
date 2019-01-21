# Exercise 3.2
# Апроксимация непрерывной функции, 2 слоя
from __future__ import print_function
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

import lab0.src.dataset5 as dataset5
from lib.custom import EarlyStoppingByLossVal, custom_fit
import lib.gui_reporter as gr
from sklearn.utils import shuffle

if __name__ == '__main__':
    # 1 parameters initializing---------------------------------------------------------
    np.random.seed(18)
    train_size = 48000
    batch_size = 480
    epochs = 300
    lr = 0.1
    goal_loss = 0.0001
    neurons_number = [80, 40]

    optimizer = Adam(lr=lr, decay=0.0001)
    loss = 'mse'

    draw_step = 10
    verbose = 1

    # 2 model and data initializing---------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False)

    x_train = np.transpose(np.append(x_train, np.ones(x_train.size)).reshape(2, x_train.size))
    x_test = np.transpose(np.append(x_test, np.ones(x_test.size)).reshape(2, x_test.size))

    model = Sequential()

    model.add(Dense(neurons_number[0], input_dim=2, activation='sigmoid'))
    model.add(Dense(neurons_number[1], activation='sigmoid'))

    model.add(Dense(1, activation='linear'))

    # 3 setting stopper---------------------------------------------------------
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=0)]

    model.compile(optimizer=optimizer, loss=loss)

    # 4 model fitting---------------------------------------------------------

    dir_name = None

    compare_title = 'approximation comparison\nlr = %.3f\n neurons = %.d %.d' % \
                    (lr, neurons_number[0], neurons_number[1])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test), )

    plt.plot(np.transpose(x_test)[0], y_test, '.')
    plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.')
    plt.legend(('function', 'approximation'), loc='upper left', shadow=True)
    plt.title(
        compare_title + "\nval_loss = %.4f\nepoch = %d" % (history.history["val_loss"][history.epoch.__len__() - 1],
                                                           epochs))

    plt.show()
    plt.close()

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save=False, show=True)
