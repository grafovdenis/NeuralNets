# Exercise №2

import sys

sys.path.append("../..")

from lab0.src.dataset5 import load_data
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

import lib.gui_reporter as gr

if __name__ == '__main__':
    # 2.1.1-------------------------------------------------------
    # Апроксимация нелинейной зависимости
    train_size = 100
    (x_train, y_train), (x_test, y_test) = load_data(train_size, show=True, k=1, b=0, func_type='n_lin')

    # 2.1.2-------------------------------------------------------
    #
    first_layer_nur = 1
    lr = 0.3
    batch_size = 10
    epochs = 10
    verbose = 1

    model = Sequential()

    model.add(Dense(first_layer_nur, input_dim=1, kernel_initializer='glorot_normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['mae'])

    # 2.1.3-------------------------------------------------------
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    score = model.evaluate(x_test, y_test, verbose=verbose)

    print("\nabsolute_error on train data\t %.f%%" % (history.history['mean_absolute_error'][epochs - 1] * 100))
    print("\nabsolute_error on testing data %.f%%" % (score[1] * 100))
    print("loss on train data %.f%%" % (history.history['loss'][epochs - 1] * 100))

    gr.plot_graphic(x=history.epoch, y=np.array(history.history['loss']),
                    x_label='epochs', y_label='loss', title='mean_squared_error history', show=True)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history['mean_absolute_error']),
                    x_label='epochs', y_label='accuracy', title='mean_absolute_error history', show=True)

    plt.plot(np.append(x_train, x_test), model.predict(np.append(x_train, x_test)), '.')
    plt.plot(np.append(x_train, x_test), np.append(y_train, y_test), '.')

    plt.legend(('approximation', 'function'), loc='upper left', shadow=True)

    plt.show()
    plt.close()

    # 2.2-------------------------------------------------------
    # Выбор скорости обучения

    h = 0.05
    max_lr = 0
    for max_lr in np.arange(max_lr, 1.0, h, dtype=float):
        print("\nrate = ", max_lr)
        loc_loss = history.history['loss'][epochs - 1]
        model = Sequential()

        model.add(Dense(first_layer_nur, input_dim=1, kernel_initializer='glorot_normal', activation='linear'))

        model.compile(loss='mean_squared_error', optimizer=SGD(lr=max_lr), metrics=['mae'])

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

        score = model.evaluate(x_test, y_test, verbose=0)

        print("loss on train data %.f%%" % (history.history['loss'][epochs - 1] * 100))

        if history.history['loss'][epochs - 1] - loc_loss >= 0.01:
            break
    max_lr -= h

    # последовательность экспериментов для трех вариантов (скорость
    # обучения а) больше (1 значение), б) равна (1 значение), в) меньше максимально допустимой
    titles = np.array(['lower than max', 'max', 'higher than max'])
    title_i = 0

    step = 0.05

    for max_lr in np.arange(max_lr - 3 * step, max_lr + 4 * step, 3 * step, dtype=float):

        model = Sequential()

        weights_arr = np.array([[2.0]], dtype=float)
        bias_arr = np.array([-0.25], dtype=float)
        x_range = np.array([1])

        model.add(Dense(first_layer_nur, input_dim=1, weights=list([weights_arr, bias_arr]), activation='linear'))

        model.compile(loss='mean_squared_error', optimizer=SGD(lr=max_lr), metrics=['mae'])

        for i in np.arange(1, epochs, 1, dtype=float):
            model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
            weights_arr = np.append(weights_arr, model.get_weights()[0])
            bias_arr = np.append(bias_arr, model.get_weights()[1])
            x_range = np.append(x_range, i)

        plt.plot(x_range, weights_arr, '.')
        plt.plot(x_range, bias_arr, '.')
        plt.title('learning rate = %-0.2f %s' % (max_lr, titles[title_i]))
        plt.legend(('weights', 'biases'), loc='upper left', shadow=True)

        plt.show()
        plt.close()
        title_i += 1
