import numpy as np

import detorch
from detorch import function
from .maths import broadcast_to
import detorch.functional as F


# TODO: implement reduction mechanism here.
# btw, the reduction is all implemented here by mean. get rid of it and let it as it is before the reduciton
class MeanSquareError(function.Function):
  def __init__(self, reduction=F.mean):
    super(MeanSquareError, self).__init__()
    self.reduction = reduction
  
  def forward(self, input, target):
    return np.mean(np.square(input - target))
  
  def backward(self, dy):
    input, target = self.inputs
    diff = input - target
    dy = broadcast_to(dy, diff.shape)
    dy = dy * diff * 2 / diff.size
    return dy, -dy


class NLLLoss1d(function.Function):
  """1d nll loss function. the input shape is 2d array (N, C), and nll loss is done on the C axis (horizontal).
  In turn, target shape is 1d array (N) whose values are integer (0 ~ C - 1)
  representing the index of the class in C classification.
  """
  
  def __init__(self, reduction=F.mean):
    super(NLLLoss1d, self).__init__()
    self.reduction = reduction
    self.min = 1e-15
    self.max = 1.0
  
  def forward(self, input, target):
    input = np.clip(input, self.min, self.max)
    indices = np.arange(input.shape[0])
    target = target.squeeze()
    return np.mean(-np.log(input[indices, target]))
  
  def backward(self, dy):
    input, target = self.inputs
    dy = dy / target.shape[0]

    # on raw data (numpy)
    indices = np.arange(input.shape[0])
    target = target.data.squeeze()
    dy_input = np.zeros(input.shape)
    dy_input[indices, target] = -1 / input.data[indices, target]  # derivative of -log(input)
    dy_input = dy_input * (input.data >= self.min) * (input.data <= self.max)  # derivative of clip

    return detorch.Tensor(dy_input) * dy, None # target is the label data


# TODO: fix the problem with test_classification.py
class CrossEntropyLoss1d(function.Function):
  def __init__(self, reduction=F.mean):
    super(CrossEntropyLoss1d, self).__init__()
    self.reduction = reduction
    self.min = 1e-15
    self.max = 1.0
    self._input = None
  
  def forward(self, input, target):
    # softmax
    c = np.mean(input, axis=-1, keepdims=True)
    input = np.exp(input - c)
    input = input / np.sum(input, axis=-1, keepdims=True)
    self._input = input  # the value after softmax will be used in back prop.
    
    # nll (1d)
    indices = np.arange(input.shape[0])
    input = np.clip(input, self.min, self.max)
    return np.mean(-np.log(input[indices, target]))
  
  def backward(self, dy):
    input = self._input.copy()  # CAUTION: self._input is numpy
    _, target = self.inputs
    dy = dy / input.shape[0]  # since reduction is now set to mean
    
    # numpy operation
    target = target.data.squeeze()
    indices = np.arange(input.shape[0])
    mask = (input >= self.min) * (input <= self.max)
    input[indices, target] -= 1.0
    input *= mask

    return detorch.Tensor(input) * dy, None  # target is the label data


def mse_loss(input, target, reduction=F.mean):
  return MeanSquareError(reduction)(input, target)


def msa_loss(input, target, reduction=F.mean):
  return reduction(F.abs(input - target))


def nll_loss(input, target, reduction=F.mean):
  # The first row is for min-batch
  if input.ndim == 2:
    return NLLLoss1d(reduction)(input, target)
  else:
    raise NotImplementedError("nll for input with ndim != 2 is not yet implemented")


def cross_entropy_loss(input, target, reduction=F.mean):
  # The first row is for min-batch
  if input.ndim == 2:
    return CrossEntropyLoss1d(reduction)(input, target)
  else:
    raise NotImplementedError("nll for input with ndim != 2 is not yet implemented")


# deprecated
def _nll_loss(input, target):
  input = F.clip(input, 1e-15, 1.0)
  return F.mean(-F.log(input[range(input.shape[0]), target]))


def _cross_entropy_loss(input, target):
  input = F.softmax(input)
  return _nll_loss(input, target)

