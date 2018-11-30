# Fetch data from CIFAR-10 dataset (consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.)
import os
from keras.datasets import cifar10
import scipy.misc

import matplotlib.pyplot as plt

output_directory = '../cifar-10/'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = {'[0]': 'airplane', '[1]': 'automobile', '[2]': 'bird', '[3]': 'cat', '[4]': 'deer',
               '[5]': 'dog', '[6]': 'frog', '[7]': 'horse', '[8]': 'ship', '[9]': 'truck'}

plt.figure(figsize=(16, 16))
count = 0
airplanes = 0
automobiles = 0
birds = 0
cats = 0
dogs = 0
for i in range(200):
    if str(y_train[i]) == '[4]' or str(y_train[i]) == '[6]' or str(y_train[i]) == '[7]' or str(
            y_train[i]) == '[8]' or str(y_train[i]) == '[9]':
        continue
    else:
        if class_names[str(y_train[i])] == 'airplane':
            airplanes += 1
        if class_names[str(y_train[i])] == 'automobile':
            automobiles += 1
        if class_names[str(y_train[i])] == 'bird':
            birds += 1
        if class_names[str(y_train[i])] == 'cat':
            cats += 1
        if class_names[str(y_train[i])] == 'dog':
            dogs += 1
        count += 1
        plt.subplot(15, 7, count)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = x_train[i]
        plt.imshow(img)
        plt.xlabel(class_names[str(y_train[i])])
        print(class_names[str(y_train[i])])
        scipy.misc.imsave(os.path.join(output_directory, class_names[str(y_train[i])] + str(i) + '.png'), x_train[i])
plt.show()
print(automobiles, airplanes, birds, cats, dogs)
