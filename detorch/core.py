import contextlib
from abc import ABC, abstractmethod
import weakref
import numpy as np
from ds import Heap

DEFAULT_DTYPE = np.float64


class Config:
    enable_backprop = True


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
    def __init__(self, data, name='', dtype=DEFAULT_DTYPE, parent_f=None):
        self.data = self.as_array(data, dtype)
        self.name = name
        self.grad = None
        self._parent_f = parent_f
        self.gen = 0 if parent_f is None else parent_f.gen + 1  # generation for graph computation order

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

    # TODO: compute self.grad with Tensor, now self.grad is np.ndarray
    def backward(self, retain_grads=False, create_graph=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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
    def as_array(data, dtype=DEFAULT_DTYPE):
        if type(data) == np.ndarray:
            return data.astype(dtype)
        elif np.isscalar(data):
            return np.array(data, dtype=dtype)
        elif isinstance(data, (tuple, list)):
            return np.array(data, dtype=dtype)
        else:
            raise TypeError(f"data type of {data} is not compatible. data should be an array-like data structure")


class Function(ABC):
    def __call__(self, *inputs):
        inputs = [self.as_variable(i) for i in inputs]
        xs = [i.data for i in inputs]
        xs = self.forward(*xs)
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
    def as_variable(x, dtype=DEFAULT_DTYPE):
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
        self.old_shape = ()
        self.shape = shape

    def forward(self, x):
        self.old_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, dy):
        return dy.reshape(self.old_shape)


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
        x0, x1 = self.inputs
        return (x1 * dy, x0 * dy)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, dy):
        x0, x1 = self.inputs
        return (dy / x1, dy * -x0 / x1 ** 2)


class Pow(Function):
    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, x):
        return x ** self.exponent

    def backward(self, dy):
        x, = self.inputs
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
        x, = self.inputs
        return np.exp(x) * dy


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, dy):
        x, = self.inputs
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


def pow(input1, exponent):
    return Pow(exponent)(input1)


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



