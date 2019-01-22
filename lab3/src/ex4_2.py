from sklearn import metrics
from neupy import algorithms

import lab0.src.dataset8 as dataset8

print(
    "\n\nРешим задачу классификации многомерных образов с помощью PNN используя обучающие выборки разных размеров\n\n")

train_size = [15000, 10000, 5000, 2000]
std = [2, 1, 0.5, 0.25]

for j in range(0, 4):
    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=0)

    print("train_size = %.d, spread = %.2f" % (train_size[j], std[j]))

    pnn = algorithms.PNN(std=std[j], batch_size=128, verbose=False)

    pnn.train(x_train[0:train_size[j]], y_train[0:train_size[j]])

    y_predicted = pnn.predict(x_test)

    maes = metrics.accuracy_score(y_predicted, y_test)

    print("accuracy = %.2f" % (maes))
