import detorch.functional as F


def sigmoid(x):
    return 1 / (1 + F.exp(-x))


def relu(x):
    return F.max(0, x)

