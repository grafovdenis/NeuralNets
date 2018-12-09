import keras.backend


def hard_lim(x):
    return keras.backend.cast(keras.backend.greater_equal(x, 0), keras.backend.floatx())
