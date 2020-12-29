from abc import ABC, abstractmethod
import numpy as np
from variable import Variable


class Function(ABC):
    def __call__(self, input):
        self.input = input
        x = input.data
        x = self.forward(x)
        self.output = Variable(x, parent=self)
        return self.output

    @abstractmethod
    def forward(self, x):
        """"""

    @abstractmethod
    def backward(self, dy):
        """"""


def Square(input=None):
    if input is not None:
        return _Square()(input)
    return _Square()


def Exp(input=None):
    if input is not None:
        return _Exp()(input)
    return _Exp()


class _Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, dy):
        x = self.input.data
        return 2 * x * dy


class _Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dy):
        x = self.input.data
        return np.exp(x) * dy

