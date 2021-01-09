from detorch import Tensor
import detorch.functional as F


def sigmoid(x: Tensor):
    return 1 / (1 + F.exp(-x))


def relu(x: Tensor):
    return F.max(0, x)

