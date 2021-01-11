from abc import ABC, abstractmethod
import weakref
from .configuration import Config
from detorch import tensor


class Function(ABC):
    def __call__(self, *inputs):
        inputs = [tensor.Tensor.as_tensor(i, dtype=Config.default_dtype) for i in inputs]
        xs = [i.data for i in inputs]

        xs = self.forward(*xs)  # all x in xs is supposed to be np.ndarray
        if not isinstance(xs, tuple):
            xs = (xs, )

        parent_f = self if Config.enable_backprop else None
        self.gen = max([i.gen for i in inputs]) if Config.enable_backprop else 0  # generation for graph
        ys = [tensor.Tensor(x, parent_f=parent_f) for x in xs]

        if Config.enable_backprop:
            self.inputs = inputs
            self.outputs = [weakref.ref(y) for y in ys]

        return ys if len(ys) > 1 else ys[0]

    @abstractmethod
    def forward(self, *xs):
        """"""

    @abstractmethod
    def backward(self, dy):
        """"""


