# Task 6.3
# Классификация многомерных образов зашумленных образов
from __future__ import print_function
from keras_preprocessing import image
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys

sys.path.append("../..")

import lab0.src.dataset8 as dataset8
import lib.gui_reporter as gr
from lib.custom import EarlyStoppingByLossVal

if __name__ == '__main__':
    np.random.seed(18)
    # 1,2 initializing
    batch_size = 32
    num_classes = 10
    epochs = 10
    lr = 0.01
    verbose = 1
    neurons_number = [256, num_classes]

    opt_name = "Adam"
    optimizer = Adam()

    goal_loss = 0.013

    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=2, show=True, show_indexes=[0, 1])

    model = Sequential()

    model.add(Dense(neurons_number[0], input_dim=28 ** 2, activation='relu'))

    model.add(Dense(neurons_number[1], activation='softmax'))

    # 3 setting stopper
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

    # 4 model fitting
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test), shuffle=True)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save=False, show=True)

    score = model.evaluate(x_test, y_test, verbose=1)
    print("accuracy on testing data %.f%%" % (score[1] * 100))

    print("Test on images drawn with Photoshop")
    local_path = 'my_images/'
    fig = plt.figure(figsize=(5, 5))
    accuracy = 0
    for im in np.arange(0, 10):
        img = image.load_img(local_path + str(im) + '.png', target_size=(28, 28), color_mode="grayscale")
        fig.add_subplot(5, 2, im + 1)
        plt.imshow(img)

        # convert to numpy array
        x = image.img_to_array(img)

        # Inverting and normalizing image
        x = 255 - x
        x /= 255
        x = np.expand_dims(x, axis=0)
        x.shape = (1, 784)

        prediction = model.predict(x)
        if np.argmax(prediction) == im:
            accuracy += 10

        print('REAL \t\tPREDICTED')
        print(str(im) + '\t\t\t' + str(np.argmax(prediction)))
    print('Accuracy on own data:', accuracy, '%')
    plt.show()
