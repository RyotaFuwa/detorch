import detorch.functional as F


def sigmoid(x):
  return 1 / (1 + F.exp(-x))


def relu(x):
  return F.max(0, x)


def softmax(x):
  c = F.mean(x, axis=-1, keepdims=True)
  y = F.exp(x - c)
  return y / F.sum(y, axis=-1, keepdims=True)
