import matplotlib.pyplot as plt
import random as rand


def draw_rectangle(x, y, x1, y1, x2, y2):
    return (x1 < x < x2) & (y1 < y < y2)


pointNumber = 1000
xCoordinates = []
yCoordinates = []

for i in range(pointNumber):
    x = rand.uniform(0, 1)
    y = rand.uniform(0, 1)
    while not draw_rectangle(x, y, 0.7, 0.2, 0.8, 0.8) | draw_rectangle(x, y, 0.1, 0.2, 0.2, 0.8) \
              | draw_rectangle(x, y, 0.2, 0.4, 0.4, 0.6) | draw_rectangle(x, y, 0.4, 0.2, 0.5, 0.8):
        x = rand.uniform(0, 1)
        y = rand.uniform(0, 1)
    xCoordinates.append(x)
    yCoordinates.append(y)

xCoordinates = list(map(lambda x: x, xCoordinates))
yCoordinates = list(map(lambda y: y, yCoordinates))
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.plot(xCoordinates, yCoordinates, '.')

plt.savefig('../images/second.png')

plt.show()
