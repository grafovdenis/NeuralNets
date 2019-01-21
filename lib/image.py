import numpy as np


def deform_image(arr, shape, k, n, m):
    if n > m:
        c = n
        n = m
        m = c
    A = arr.shape[0] / 3.0
    w = 2.0 / arr.shape[1]
    shift = lambda x: A * np.sin(2.0 * np.pi * x * w)
    for i in range(n, m):
        arr[:, i] = np.roll(arr[:, i], int(shift(i) * k))
    return arr.reshape(shape)


def noise(arr, intensity):
    for i in range(1, arr.size):
        i += intensity * np.random.randint(0, 3)
        if i > 780:
            i = 780
        if arr[i] > 0.5:
            arr[i] = 0.0
        else:
            arr[i] = 1.0
    return arr


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst
