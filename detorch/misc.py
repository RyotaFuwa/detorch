import numpy as np
from .core import Tensor


# tensor alternatives for numpy functions
def zeros_like(x: Tensor):
    data = np.zeros_like(x.data)
    return Tensor(data)


def ones_like(x: Tensor):
    data = np.ones_like(x.data)
    return Tensor(data)


def randn_like(x: Tensor):
    data = np.random.randn(*x.shape)
    return data


def zeros(*shape):
    data = np.zeros(shape)
    return Tensor(data)


def ones(*shape):
    data = np.ones(shape)
    return Tensor(data)


def randn(*shape):
    data = np.random.randn(*shape)
    return Tensor(data)


def fill(num, shape):
    data = np.ones(shape) * num
    return Tensor(data)
