import numpy as np
from variable import Variable
from function import *


def test_autograd_1():
    input = Variable(np.array(0.5))
    x = Square(input)
    x = Exp(x)
    output = Square(x)

    output.grad = np.array(1.0)
    output.backward()
    print(input.grad)


def test_autograd_2():
    input = Variable(np.array(0.5))
    x = Square()(input)
    x = Exp()(x)
    output = Square()(x)

    output.grad = np.array(1.0)
    output.backward()
    print(input.grad)


if __name__ == '__main__':
    test_autograd_2()
