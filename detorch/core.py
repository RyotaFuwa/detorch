import contextlib
from abc import ABC, abstractmethod
import weakref
import numpy as np
from ds import Heap
from .utils import *


class Config:
    enable_backprop = True
    default_dtype = np.float64


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


@contextlib.contextmanager
def no_grad():
    return using_config('enable_backprop', False)


# TODO: topological sort for graph computation
class Tensor:
    __array_priority__ = 1

    def __init__(self, data, name='', dtype=None, parent_f=None):
        self.data = self.as_array(data, dtype)
        self.name = name
        self.grad = None
        self._parent_f = parent_f
        self.gen = 0 if parent_f is None else parent_f.gen + 1  # generation for graph computation order

    def copy(self):
        r"""copy the tensor. Technically, this operation copies only data and DOES NOT copy parent_f"""
        return Tensor(self.data.copy(), name=self.name, parent_f=self.parent_f)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    @property
    def parent_f(self):
        return self._parent_f

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return transpose(self)

    # TODO: compute self.grad with Tensor, now self.grad is np.ndarray
    def backward(self, retain_grads=False, create_graph=False):
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))

        heap = Heap(key=lambda x: -x.gen)
        heap.push(self._parent_f)
        nodes_seen = set()
        while len(heap) > 0:
            f = heap.pop()
            dys = [y().grad for y in f.outputs]
            with using_config('enable_backprop', create_graph):
                grads = f.backward(*dys)
                if not isinstance(grads, tuple):
                    grads = (grads, )

                for i, grad in zip(f.inputs, grads):
                    if i.grad is not None:
                        i.grad = i.grad + grad  # CAUTION: do not use +=
                    else:
                        i.grad = grad

                    if i.parent_f is not None and i.parent_f not in nodes_seen:
                        nodes_seen.add(i.parent_f)
                        heap.push(i.parent_f)

                if not retain_grads:
                    for y in f.outputs:
                        y().grad = None

    def zero_grad(self):
        self.grad = None
        if self.parent_f is None:
            return
        f_nodes = [self.parent_f]
        while f_nodes:
            f = f_nodes.pop()
            for i in f.inputs:
                i.grad = None
                if i.parent_f is not None:
                    f_nodes.append(i.parent_f)

    def view(self, *shape):
        return view(self, *shape)

    def view_(self, *shape):
        self.data.reshape(shape)
        return self

    def transpose(self):
        return transpose(self)

    def transpose_(self):
        self.data = self.data.T
        return self

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)

    def add(self, other):
        return add(self, other)

    def add_(self, other):
        self.data += other.data
        return self

    def sub(self, other):
        return sub(self, other)

    def sub_(self, other):
        self.data -= other.data
        return self

    def mul(self, other):
        return mul(self, other)

    def mul_(self, other):
        self.data -= other.data
        return self

    def div(self, other):
        return div(self, other)

    def div_(self, other):
        self.data -= other.data
        return self

    def __neg__(self):
        return neg(self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, power):
        return pow(self, power)

    @staticmethod
    def as_array(data, dtype=None):
        if isinstance(data, np.ndarray):
            if dtype is None:
                return data
            return data.astype(dtype)
        elif np.isscalar(data):
            return np.array(data, dtype=dtype if dtype is not None else Config.default_dtype)
        elif isinstance(data, (tuple, list)):
            return np.array(data, dtype=dtype if dtype is not None else Config.default_dtype)
        else:
            raise TypeError(f"data type of {data} is not compatible. data should be an array-like data structure")


class Function(ABC):
    def __call__(self, *inputs):
        inputs = [self.as_variable(i, dtype=Config.default_dtype) for i in inputs]
        xs = [i.data for i in inputs]

        xs = self.forward(*xs)  # all x in xs is supposed to be np.ndarray
        if not isinstance(xs, tuple):
            xs = (xs, )

        parent_f = self if Config.enable_backprop else None
        self.gen = max([i.gen for i in inputs]) if Config.enable_backprop else 0  # generation for graph
        ys = [Tensor(x, parent_f=parent_f) for x in xs]

        if Config.enable_backprop:
            self.inputs = inputs
            self.outputs = [weakref.ref(y) for y in ys]

        return ys if len(ys) > 1 else ys[0]

    @staticmethod
    def as_variable(x, dtype=None):
        if isinstance(x, Tensor):
            return x
        return Tensor(x, dtype=dtype)

    @abstractmethod
    def forward(self, *xs):
        """"""

    @abstractmethod
    def backward(self, dy):
        """"""


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


def tmp_test():
    def f(x):
        y = x ** 4 - 2 * x ** 2
        return y
    x = Tensor(2)
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad)
    gx = x.grad
    x.zero_grad()
    gx.backward()
    print(x.grad)


if __name__ == '__main__':
    tmp_test()



