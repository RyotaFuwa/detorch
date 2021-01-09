from .maths import square, mean


def mean_square_error(y, x):
    a = y - x
    b = square(a)
    c = mean(b)
    return c
    # return mean(square(y - x))
