from abc import ABC, abstractmethod
import weakref
import numpy as np
from variable import Variable
from config import Config

DEFAULT_DTYPE = np.float64


class Function(ABC):
    def __call__(self, *inputs):
        inputs = [self.as_variable(i) for i in inputs]
        xs = [i.data for i in inputs]
        xs = self.forward(*xs)
        if not isinstance(xs, tuple):
            xs = (xs, )

        parent_f = self if Config.enable_backprop else None
        self.gen = max([i.gen for i in inputs]) if Config.enable_backprop else 0  # generation for graph
        ys = [Variable(x, parent_f=parent_f) for x in xs]

        if Config.enable_backprop:
            self.inputs = inputs
            self.outputs = [weakref.ref(y) for y in ys]

        return ys if len(ys) > 1 else ys[0]

    @staticmethod
    def as_variable(x, dtype=DEFAULT_DTYPE):
        if isinstance(x, Variable):
            return x
        return Variable(x, dtype=dtype)

    @abstractmethod
    def forward(self, *xs):
        """"""

    @abstractmethod
    def backward(self, dy):
        """"""


# TODO: Type checking for inputs and outputs
# each function has requirements on the shape and type of each input and output, also the num of inputs and outputs
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, dy):
        return -dy


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, dy):
        return (dy, dy)


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, dy):
        return (dy, -dy)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, dy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return (x1 * dy, x0 * dy)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, dy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return (dy / x1, dy * -x0 / x1 ** 2)


class Pow(Function):
    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, x):
        return x ** self.exponent

    def backward(self, dy):
        x = self.inputs[0].data
        return dy * self.exponent * x ** (self.exponent - 1)


class Square(Pow):
    def __init__(self):
        super().__init__(2)


class Cube(Pow):
    def __init__(self):
        super().__init__(3)


class SquareRoot(Pow):
    def __init__(self):
        super().__init__(0.5)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dy):
        x = self.inputs[0].data
        return np.exp(x) * dy


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, dy):
        x = self.inputs[0].data
        return dy / x


def neg(input):
    return Neg()(input)


def add(input1, input2):
    return Add()(input1, input2)


def sub(input1, input2):
    return Sub()(input1, input2)


def mul(input1, input2):
    return Mul()(input1, input2)


def div(input1, input2):
    return Div()(input1, input2)


def pow(input1, input2):
    return Pow(input2)(input1)


# a lot of functions

