from keras import Sequential, callbacks
from keras.layers import Dense
from keras.optimizers import SGD
import sys

sys.path.append('../..')

from lab0.src import dataset5
import lib.gui_reporter as gr

train_size = 4000

first_layer_nur = 1
lr = 0.3
batch_size = 20
epochs = 50
verbose = 1

(x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=True, func_type='n_lin', k=1,
                                                          b=0.1)

model = Sequential()

model.add(Dense(1, kernel_initializer='glorot_normal', activation='hard_sigmoid'))

# early stop
stopper = callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, mode='max')

model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['accuracy'])

# batch_size define speed of studying
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[stopper], verbose=verbose)

score = model.evaluate(x_test, y_test, verbose=1)

print("Accuracy on train data\t %.f%%" % (history.history['acc'][stopper.stopped_epoch] * 100))
print("Accuracy on testing data %.f%%" % (score[1] * 100))
print("Loss on train data %.f%%" % (history.history['loss'][stopper.stopped_epoch] * 100))
gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png",
                        save=True, show=True)
