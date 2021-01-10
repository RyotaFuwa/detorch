import numpy as np
from detorch import Config, Parameter
from .layer import Layer
import detorch.functional as F


class Linear(Layer):
    def __init__(self, input_size, output_size, bias=True, dtype=Config.default_dtype, init_method=Config.init_method):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.bias = bias
        self.A, self.b = self.init_weights(init_method)

    def forward(self, x):
        if self.bias:
            return F.linear(x, self.A, self.b)
        else:
            return x @ self.A

    def init_weights(self, init_method):
        I, O = self.input_size, self.output_size
        b = None
        if init_method == 'xavier':
            A = Parameter(np.random.randn(I, O) * (1 / I), dtype=self.dtype)
            if self.bias:
                b = Parameter(np.zeros((1, O)))
        elif init_method == 'he':
            A = Parameter(np.random.randn(I, O) * (2 / I), dtype=self.dtype)
            if self.bias:
                b = Parameter(np.zeros((1, O)))
        else:
            raise NotImplemented(f'init_method: {init_method} is not implemented.')
        return A, b