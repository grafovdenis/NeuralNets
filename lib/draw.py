import math


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
        return False

    return int(sign1) == int(sign2) == int(sign3)
