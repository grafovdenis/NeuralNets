import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# ----------DOWNLOAD THE IMDB DATASET----------
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# => Training entries: 25000, labels: 25000


# ----------CONVERT INTEGERS TO WORDS----------
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# function to convert integers to words
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# ----------PREPARE DATA----------
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# len(train_data[0]) == len(train_data[1])


# ----------BUILD THE MODEL----------
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
# takes integer-encoded vocab and looks up the embedding vector for each word-index
# result dimensions: (batch(collection), sequence, embedding)
model.add(keras.layers.Embedding(vocab_size, 16))
# result: fixed-length output vector for each example by averaging over the sequence dimension
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# result: float 0..1
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',  # mean_squared_error in also available
              metrics=['accuracy'])

# ----------CREATE A VALIDATION SET----------
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# ----------EVALUATE THE MODEL----------
results = model.evaluate(test_data, test_labels)
print(results)

# ----------CREATE A GRAPH OF ACCURACY AND LOSS OVER TIME----------
history_dict = history.history
# print(history_dict.keys()) => dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(acc) + 1)

# Losses
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'ro', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracies
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
