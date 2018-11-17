import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot(111)

t = np.linspace(0, 5, 500)
s = np.abs(np.sin(t) * np.cos(4 * np.pi * t) * np.arctan(t - 1) - 1)
line, = plt.plot(t, s)

plt.savefig('../plots/fifth.png')

plt.show()
