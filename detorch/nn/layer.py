from abc import ABC, abstractmethod
import weakref
from detorch.parameter import Parameter


class Layer(ABC):
    def __init__(self):
        self._params = set()

    def __setattr__(self, key, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(key)
        super().__setattr__(key, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(i) for i in inputs]
        self.outputs = [weakref.ref(o) for o in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *inputs):
        """"""

    def parameters(self):
        for key in self._params:
            obj = self.__dict__[key]
            if isinstance(obj, Layer):
                yield from obj.parameters()
            else:
                yield obj

    def clear_grad(self):
        for param in self.parameters():
            param.clear_grad()

    # torch syntactical alternative for clear_grad
    def zero_grad(self):
        self.clear_grad()





