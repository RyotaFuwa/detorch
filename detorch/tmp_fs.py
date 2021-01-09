import numpy as np
from .core import Function
from .utils import *


# TODO: Type checking for inputs and outputs
# each function has requirements on the shape and type of each input and output, also the num of inputs and outputs
class View(Function):
    def __init__(self, *shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, dy):
        return view(self, *self.x_shape)


class Transpose(Function):
    def forward(self, x):
        return x.T

    def backward(self, dy):
        return transpose(dy)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, dy):
        return -dy


class Add(Function):
    def forward(self, x0, x1):
        self.shape_x0, self.shape_x1 = x0.shape, x1.shape
        return x0 + x1

    def backward(self, dy):
        dy0, dy1 = dy, dy
        if self.shape_x0 != self.shape_x1:
            dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
        return (dy0, dy1)


class Sub(Function):
    def forward(self, x0, x1):
        self.shape_x0, self.shape_x1 = x0.shape, x1.shape
        return x0 - x1

    def backward(self, dy):
        dy0, dy1 = dy, -dy
        if self.shape_x0 != self.shape_x1:
            dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
        return (dy0, dy1)


class Mul(Function):
    def forward(self, x0, x1):
        self.shape_x0, self.shape_x1 = x0.shape, x1.shape
        return x0 * x1

    def backward(self, dy):
        x0, x1 = self.inputs
        dy0, dy1 = (x1 * dy, x0 * dy)
        if self.shape_x0 != self.shape_x1:
            dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
        return dy0, dy1


class Div(Function):
    def forward(self, x0, x1):
        self.shape_x0, self.shape_x1 = x0.shape, x1.shape
        return x0 / x1

    def backward(self, dy):
        x0, x1 = self.inputs
        dy0, dy1 = (dy / x1, dy * -x0 / x1 ** 2)
        if self.shape_x0 != self.shape_x1:
            dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
        return dy0, dy1


class Pow(Function):
    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, x):
        return x ** self.exponent

    def backward(self, dy):
        x, = self.inputs
        return dy * self.exponent * x ** (self.exponent - 1)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, dy):
        if not self.keepdims:
            pass
        return broadcast_to(dy, self.x_shape)


class BroadcastTo(Function):
    def __init__(self, *shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, dy):
        return sum_to(dy, self.x_shape)


class SumTo(Function):
    def __init__(self, *shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np_sum_to(x, self.shape)

    def backward(self, dy):
        return broadcast_to(dy, self.x_shape)


class MatMul(Function):
    def forward(self, x0, x1):
        return np.dot(x0, x1)

    def backward(self, dy):
        x0, x1 = self.inputs
        dy0 = mm(dy, x1.T)
        dy1 = mm(x0.T, dy)
        return dy0, dy1


def view(input, *shape):
    return View(shape)(input)


def transpose(input):
    return Transpose()(input)


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


def pow(input, exponent):
    return Pow(exponent)(input)


def sum(input, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(input)


def broadcast_to(input, shape):
    if input.shape == shape:
        return input
    return BroadcastTo(*shape)(input)


def sum_to(input, shape):
    if input.shape == shape:
        return input
    return SumTo(*shape)(input)


def mm(input1, input2):
    return MatMul()(input1, input2)
