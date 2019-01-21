# Exercise №1
# Exploring of functions for weights and biases initializing
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense
import keras.initializers as wi


def plot_weights(weights, x, title):
    plt.plot(x, weights[0].reshape(x.shape), '.')
    plt.plot(x, weights[1].reshape(x.shape), '.')
    plt.legend(('weigths', 'biases'), loc='upper right', shadow=True)
    plt.title(title)
    plt.show()
    plt.close()
    return 0


if __name__ == '__main__':
    count = 1000
    input_dim = 1
    seed = 13

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim, kernel_initializer=wi.Zeros(), bias_initializer=wi.Zeros()))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='Zeros')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim, kernel_initializer=wi.Ones(), bias_initializer=wi.Ones()))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='Ones')

    model = Sequential()
    model.add(
        Dense(count, input_dim=input_dim, kernel_initializer=wi.Constant(value=3.0),
              bias_initializer=wi.Constant(value=3.0)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='Constant(value=3.0)')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim, kernel_initializer=wi.RandomNormal(mean=0.0, stddev=0.1, seed=seed),
                    bias_initializer=wi.RandomNormal(mean=0.0, stddev=0.1, seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1),
                 title='RandomNormal(mean=0.0, stddev=0.1, seed=seed)')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim, kernel_initializer=wi.RandomUniform(minval=-0.10, maxval=0.10, seed=seed),
                    bias_initializer=wi.RandomUniform(minval=-0.10, maxval=0.10, seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1),
                 title='RandomUniform(minval=-0.10, maxval=0.10, seed=seed)')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim, kernel_initializer=wi.TruncatedNormal(mean=0.0, stddev=0.1, seed=seed),
                    bias_initializer=wi.TruncatedNormal(mean=0.0, stddev=0.1, seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1),
                 title='TruncatedNormal(mean=0.0, stddev=0.1, seed=seed)')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim,
                    kernel_initializer=wi.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed),
                    bias_initializer=wi.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='VarianceScaling')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim,
                    kernel_initializer=wi.lecun_normal(seed=seed),
                    bias_initializer=wi.lecun_normal(seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='lecun_normal')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim,
                    kernel_initializer=wi.lecun_uniform(seed=seed),
                    bias_initializer=wi.lecun_uniform(seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='lecun_uniform')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim,
                    kernel_initializer=wi.glorot_normal(seed=seed),
                    bias_initializer=wi.glorot_normal(seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='glorot_normal')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim,
                    kernel_initializer=wi.glorot_uniform(seed=seed),
                    bias_initializer=wi.glorot_uniform(seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='glorot_uniform')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim,
                    kernel_initializer=wi.he_normal(seed=seed),
                    bias_initializer=wi.he_normal(seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='he_normal')

    model = Sequential()
    model.add(Dense(count, input_dim=input_dim,
                    kernel_initializer=wi.he_uniform(seed=seed),
                    bias_initializer=wi.he_uniform(seed=seed)))
    plot_weights(weights=model.get_weights(), x=np.arange(0, count, 1), title='he_uniform')
