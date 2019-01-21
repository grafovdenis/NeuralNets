# Eight dataset
import keras
import matplotlib.pyplot as plt
import sys

sys.path.append("../..")

from lib.image import *


def load_data(mode=0, show=False, show_indexes=[0]):
    """
    mode = 0 - normal images
    mode = 1 - deformed images
    mode = 2 - noised images
    """
    if not [0, 1, 2].__contains__(mode):
        Exception("Unexpected mode value")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    if mode == 1:
        for i in range(0, 60000):
            x_train[i] = deform_image(x_train[i], (28, 28), 0.3, 0, 28)
        for i in range(0, 10000):
            x_test[i] = deform_image(x_test[i], (28, 28), 0.3, 0, 28)

    resolution = x_train.shape[1]

    x_train = x_train.reshape(x_train.shape[0], resolution ** 2)
    x_test = x_test.reshape(x_test.shape[0], resolution ** 2)

    if mode == 2:
        for i in range(0, 60000):
            x_train[i] = noise(x_train[i], 2000)
        for i in range(0, 10000):
            x_test[i] = noise(x_test[i], 2000)

    # normalize inputs from 0-255 to 0.0-1.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if show:
        for i in show_indexes:
            plt.imshow(x_train[i].reshape(resolution, resolution))
            plt.show()
            plt.close()

    return (x_train, y_train), (x_test, y_test)
