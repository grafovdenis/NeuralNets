import random
from sympy import *


# a
h = 5
d = 1
n = 10
x = symarray('x', n + 1)


def y(n):
    return sum([x[n - i] * random.randrange(1, 10) for i in range(0, h)])


print("y(n) =", y(n))