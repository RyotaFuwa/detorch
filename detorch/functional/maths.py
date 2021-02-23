import numpy as np

import detorch
from detorch import function
from detorch.utils import *


# TODO: Type checking for inputs and outputs
# each function has requirements on the shape and type of each input and output, also the num of inputs and outputs
class View(function.Function):
  def __init__(self, *shape):
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    return x.reshape(self.shape)
  
  def backward(self, dy):
    return view(self, *self.x_shape)


class Clip(function.Function):
  def __init__(self, min=None, max=None):
    self.min = min
    self.max = max
  
  def forward(self, x):
    return np.clip(x, self.min, self.max)
  
  def backward(self, dy):
    """pass dy through if the corresponding x wasn't clipped"""
    x, = self.inputs
    return dy * (x >= self.min) * (x <= self.max)


class GetItem(function.Function):
  def __init__(self, slice):
    self.slice = slice
  
  def forward(self, x):
    return x[self.slice]
  
  def backward(self, dy):
    x, = self.inputs
    return _GetItemGrad(self.slice, x.shape)(dy)


class _GetItemGrad(function.Function):
  def __init__(self, slice, x_shape):
    self.slice = slice
    self.x_shape = x_shape
  
  def forward(self, x):
    dy = np.zeros(self.x_shape)
    dy[self.slice] = x
    return dy
  
  def backward(self, dy):
    return get_item(dy, self.slice)


class Transpose(function.Function):
  def forward(self, x):
    return x.T
  
  def backward(self, dy):
    return transpose(dy)


class Neg(function.Function):
  def forward(self, x):
    return -x
  
  def backward(self, dy):
    return -dy


class Abs(function.Function):
  def forward(self, x):
    return np.abs(x)
  
  def backward(self, dy):
    x, = self.inputs
    return -dy * (x < 0) + dy * (x >= 0)


class Add(function.Function):
  def forward(self, x0, x1):
    self.shape_x0, self.shape_x1 = x0.shape, x1.shape
    return x0 + x1
  
  def backward(self, dy):
    dy0, dy1 = dy, dy
    if self.shape_x0 != self.shape_x1:
      dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
    return (dy0, dy1)


class Sub(function.Function):
  def forward(self, x0, x1):
    self.shape_x0, self.shape_x1 = x0.shape, x1.shape
    return x0 - x1
  
  def backward(self, dy):
    dy0, dy1 = dy, -dy
    if self.shape_x0 != self.shape_x1:
      dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
    return (dy0, dy1)


class Mul(function.Function):
  def forward(self, x0, x1):
    self.shape_x0, self.shape_x1 = x0.shape, x1.shape
    return x0 * x1
  
  def backward(self, dy):
    x0, x1 = self.inputs
    dy0, dy1 = (x1 * dy, x0 * dy)
    if self.shape_x0 != self.shape_x1:
      dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
    return dy0, dy1


class Div(function.Function):
  def forward(self, x0, x1):
    self.shape_x0, self.shape_x1 = x0.shape, x1.shape
    return x0 / x1
  
  def backward(self, dy):
    x0, x1 = self.inputs
    dy0, dy1 = (dy / x1, dy * -x0 / x1 ** 2)
    if self.shape_x0 != self.shape_x1:
      dy0, dy1 = (sum_to(dy0, self.shape_x0), sum_to(dy1, self.shape_x1))
    return dy0, dy1


class Pow(function.Function):
  def __init__(self, exponent):
    self.exponent = exponent
  
  def forward(self, x):
    return x ** self.exponent
  
  def backward(self, dy):
    x, = self.inputs
    return dy * self.exponent * x ** (self.exponent - 1)


class Sum(function.Function):
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


class BroadcastTo(function.Function):
  def __init__(self, *shape):
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    return np.broadcast_to(x, self.shape)
  
  def backward(self, dy):
    return sum_to(dy, self.x_shape)


class SumTo(function.Function):
  def __init__(self, *shape):
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    return np_sum_to(x, self.shape)
  
  def backward(self, dy):
    return broadcast_to(dy, self.x_shape)


class MatMul(function.Function):
  def forward(self, x0, x1):
    return np.dot(x0, x1)
  
  def backward(self, dy):
    x0, x1 = self.inputs
    dy0 = mm(dy, x1.T)
    dy1 = mm(x0.T, dy)
    return dy0, dy1


class Exp(detorch.function.Function):
  def forward(self, x):
    return np.exp(x)
  
  def backward(self, dy):
    x, = self.inputs
    return exp(x) * dy


class Log(detorch.function.Function):
  def forward(self, x):
    return np.log(x)
  
  def backward(self, dy):
    x, = self.inputs
    return dy / x


class Sin(detorch.function.Function):
  def forward(self, x):
    return np.sin(x)
  
  def backward(self, dy):
    x, = self.inputs
    return cos(x) * dy


class Cos(detorch.function.Function):
  def forward(self, x):
    return np.cos(x)
  
  def backward(self, dy):
    x, = self.inputs
    return -sin(x) * dy


class Tan(detorch.function.Function):
  def forward(self, x):
    return np.tan(x)
  
  def backward(self, dy):
    x = self.inputs
    return dy / cos(x) ** 2


class Tanh(detorch.function.Function):
  def forward(self, x):
    return np.tanh(x)
  
  def backward(self, dy):
    y = self.outputs[0]()
    return dy * (1 - y ** 2)


class Max(detorch.function.Function):
  def forward(self, x0, x1):
    return np.maximum(x0, x1)
  
  def backward(self, dy):
    x0, x1 = self.inputs
    mask0 = x0 >= x1
    mask1 = x0 < x1
    return mask0 * dy, mask1 * dy


def view(input, *shape):
  return View(shape)(input)


def clip(input, min, max):
  return Clip(min, max)(input)


def get_item(input, slice):
  return GetItem(slice)(input)


def transpose(input):
  return Transpose()(input)


def neg(input):
  return Neg()(input)


def abs(input):
  return Abs()(input)


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


def linear(x, A, b):
  return x @ A + b


def exp(input):
  return Exp()(input)


def log(input):
  return Log()(input)


def sin(input):
  return Sin()(input)


def cos(input):
  return Cos()(input)


def tan(input):
  return Tan()(input)


def tanh(input):
  return Tanh()(input)


def max(input0, input1):
  return Max()(input0, input1)


def square(input):
  return pow(input, 2)


def cube(input):
  return pow(input, 3)


def sqrt(input):
  return pow(input, 0.5)


def sphere(x, y):
  return x ** 2 + y ** 2


def matyas(x, y):
  return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
  return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
         (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))


def rosenbrock(x0, x1):
  return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2


# statistics
def mean(x, axis=None, keepdims=False):
  return x.sum(axis=axis, keepdims=keepdims) / x.size


def tmp_test():
  x = detorch.Tensor(1)
  y = detorch.Tensor(1)
  z = sphere(x, y)
  z.backward()
  print(x.grad, y.grad)
  z.zero_grad()
  
  z = matyas(x, y)
  z.backward()
  print(x.grad, y.grad)
  z.zero_grad()
  
  z = goldstein(x, y)
  z.backward()
  print(x.grad, y.grad)


if __name__ == '__main__':
  tmp_test()
