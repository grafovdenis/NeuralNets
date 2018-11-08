import math

import matplotlib.pyplot as plt
import random as rand


def draw_rectangle(x, y, x1, y1, x2, y2):
    return (x1 < x < x2) & (y1 < y < y2)


def draw_circle(x, y, x1, y1, r):
    return not (x1 - x) ** 2 + (y1 - y) ** 2 < r ** 2


def draw_triangle(x, y, x1, y1, x2, y2, x3, y3):
    sign1 = (x1 - x) * (y2 - y1) - (x2 - x1) * (y1 - y)
    sign2 = (x2 - x) * (y3 - y2) - (x3 - x2) * (y2 - y)
    sign3 = (x3 - x) * (y1 - y3) - (x1 - x3) * (y3 - y)

    # Нормализация
    try:
        sign1 /= math.fabs(sign1)
        sign2 /= math.fabs(sign2)
        sign3 /= math.fabs(sign3)
    except ZeroDivisionError:
        return True

    return not int(sign1) == int(sign2) == int(sign3)


pointNumber = 3000
xCoordinates = []
yCoordinates = []

# B
for i in range(pointNumber):
    x = rand.uniform(0, 1)
    y = rand.uniform(0, 2)
    while not draw_rectangle(x, y, 0.1, 1.2, 0.2, 1.8) | draw_rectangle(x, y, 0.35, 1.2, 0.4, 1.8) \
              | draw_rectangle(x, y, 0.2, 1.45, 0.4, 1.55) | draw_rectangle(x, y, 0.2, 1.7, 0.4, 1.8) \
              | draw_rectangle(x, y, 0.2, 1.2, 0.4, 1.3):
        x = rand.uniform(0, 1)
        y = rand.uniform(0, 2)
    xCoordinates.append(x)
    yCoordinates.append(y)

# E
for i in range(pointNumber):
    x = rand.uniform(0, 1)
    y = rand.uniform(0, 2)
    while not draw_rectangle(x, y, 0.5, 1.2, 0.6, 1.8) \
              | draw_rectangle(x, y, 0.6, 1.45, 0.8, 1.55) \
              | draw_rectangle(x, y, 0.6, 1.7, 0.8, 1.8) \
              | draw_rectangle(x, y, 0.6, 1.2, 0.8, 1.3):
        x = rand.uniform(0, 1)
        y = rand.uniform(0, 2)
    xCoordinates.append(x)
    yCoordinates.append(y)

# F
for i in range(pointNumber):
    x = rand.uniform(0, 1)
    y = rand.uniform(0, 2)
    while not draw_rectangle(x, y, 0.1, 0.2, 0.2, 0.8) \
              | draw_rectangle(x, y, 0.2, 0.45, 0.4, 0.55) \
              | draw_rectangle(x, y, 0.2, 0.7, 0.4, 0.8):
        x = rand.uniform(0, 1)
        y = rand.uniform(0, 2)
    xCoordinates.append(x)
    yCoordinates.append(y)

# R
for i in range(pointNumber):
    x = rand.uniform(0, 1)
    y = rand.uniform(0, 2)
    while not draw_rectangle(x, y, 0.5, 0.2, 0.6, 0.8) \
              | draw_rectangle(x, y, 0.75, 0.55, 0.8, 0.75) \
              | draw_rectangle(x, y, 0.75, 0.2, 0.8, 0.55) \
              | draw_rectangle(x, y, 0.6, 0.45, 0.8, 0.55) \
              | draw_rectangle(x, y, 0.6, 0.7, 0.8, 0.8):
        x = rand.uniform(0, 1)
        y = rand.uniform(0, 2)
    xCoordinates.append(x)
    yCoordinates.append(y)

# E
for i in range(pointNumber):
    x = rand.uniform(0, 2)
    y = rand.uniform(0, 2)
    while not draw_rectangle(x, y, 1.0, 0.2, 1.1, 0.8) \
              | draw_rectangle(x, y, 1.0, 0.45, 1.25, 0.55) \
              | draw_rectangle(x, y, 1.0, 0.7, 1.25, 0.8) \
              | draw_rectangle(x, y, 1.0, 0.2, 1.25, 0.3):
        x = rand.uniform(0, 2)
        y = rand.uniform(0, 2)
    xCoordinates.append(x)
    yCoordinates.append(y)

# E
for i in range(pointNumber):
    x = rand.uniform(0, 2)
    y = rand.uniform(0, 2)
    while not draw_rectangle(x, y, 1.5, 0.2, 1.6, 0.8) \
              | draw_rectangle(x, y, 1.5, 0.45, 1.75, 0.55) \
              | draw_rectangle(x, y, 1.5, 0.7, 1.75, 0.8) \
              | draw_rectangle(x, y, 1.5, 0.2, 1.75, 0.3):
        x = rand.uniform(0, 2)
        y = rand.uniform(0, 2)
    xCoordinates.append(x)
    yCoordinates.append(y)

# triangle
for i in range(pointNumber):
    x = rand.uniform(0, 2)
    y = rand.uniform(0, 2)
    while draw_circle(x, y, 1.5, 1.5, 0.3) | (not draw_triangle(x, y, 1.3, 1.6, 1.6, 1.6, 1.6, 1.3)):
        x = rand.uniform(0, 2)
        y = rand.uniform(0, 2)
    xCoordinates.append(x)
    yCoordinates.append(y)

xCoordinates = list(map(lambda x: x, xCoordinates))
yCoordinates = list(map(lambda y: y, yCoordinates))
plt.xlim(0, 2)
plt.ylim(0, 2)

plt.plot(xCoordinates, yCoordinates, '.')

plt.savefig('../images/fourth.png')

plt.show()
