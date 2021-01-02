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
    def backward(self, *dys):
        """"""


def square(input):
    return Square()(input)


def exp(input):
    return Exp()(input)


def add(input1, input2):
    return Add()(input1, input2)


def mul(input1, input2):
    return Mul()(input1, input2)


# TODO: Type checking for inputs and outputs
# each function has requirements on the shape and type of each input and output, also the num of inputs and outputs
class Square(Function):
    def forward(self, *xs):
        return xs[0] ** 2

    def backward(self, dy):
        x = self.inputs[0].data
        return 2 * x * dy


class Exp(Function):
    def forward(self, *xs):
        return np.exp(xs[0])

    def backward(self, dy):
        x = self.inputs[0].data
        return np.exp(x) * dy


class Add(Function):
    def forward(self, *xs):
        x0, x1 = xs[0], xs[1]
        return x0 + x1

    def backward(self, dy):
        return (dy, dy)


class Mul(Function):
    def forward(self, *xs):
        x0, x1 = xs[0], xs[1]
        return x0 * x1

    def backward(self, dy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return (x1 * dy, x0 * dy)


def tmp_test():
    a = Variable(2.0)
    x = square(a)
    x = add(square(x), square(x))
    x.backward()
    print(a.grad)


if __name__ == '__main__':
    tmp_test()


