import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.abs(np.sin(t) * np.cos(4 * np.pi * t) * np.arctan(t - 1) - 1)
line, = plt.plot(t, s)
plt.show()
