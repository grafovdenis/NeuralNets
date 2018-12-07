# Eight dataset
# Fetch data from CIFAR-10 dataset (consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.)

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = {'[0]': 'airplane', '[1]': 'automobile', '[2]': 'bird', '[3]': 'cat', '[4]': 'deer',
               '[5]': 'dog', '[6]': 'frog', '[7]': 'horse', '[8]': 'ship', '[9]': 'truck'}

# convert class vectors to binary vectors
Y_train = keras.utils.np_utils.to_categorical(y_train)
Y_test = keras.utils.np_utils.to_categorical(y_test)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


X_train_gray = grayscale(X_train)
X_test_gray = grayscale(X_test)
print("X_train_gray shape:", X_train_gray.shape)
print("X_test_gray shape:", X_test_gray.shape)

# now we have only one channel in the images
# plot a randomly chosen image
img = 0
plt.figure(figsize=(4, 2))
print(class_names[str(y_train[img])], "displayed")
plt.subplot(1, 2, 1)
plt.imshow(X_train[img], interpolation='none')
plt.subplot(1, 2, 2)
plt.imshow(X_train_gray[img, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
plt.show()
