import numpy as np
from .maths import broadcast_to
from detorch import function


class MeanSquareError(function.Function):
    def forward(self, x0, x1):
        return np.mean(np.square(x0 - x1))

    def backward(self, dy):
        x0, x1 = self.inputs
        diff = x0 - x1
        dy = broadcast_to(dy, diff.shape)
        dy = dy * diff * 2 / diff.size
        return dy, -dy


def mean_square_error(y, x):
    return MeanSquareError()(y, x)
