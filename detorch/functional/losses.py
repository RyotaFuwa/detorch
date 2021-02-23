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
  
  def forward(self, input, target):
    input = np.clip(input, 1e-15, 1.0)
    target = target.squeeze()
    indices = np.arange(input.shape[0])
    return np.mean(-np.log(input[indices, target]))
  
  def backward(self, dy):
    input, target = self.inputs
    dy = dy / target.shape[0]
    
    # on raw data (numpy)
    indices = np.arange(input.shape[0])
    target = target.data.squeeze()
    zeros_input = np.zeros(input.shape)
    zeros_input[indices, target] -= dy.data / input[indices, target]
    ones_target = np.ones(target.shape)
    
    return detorch.Tensor(zeros_input) * dy, detorch.Tensor(ones_target) * dy


class CrossEntropyLoss1d(function.Function):
  def __init__(self, reduction=F.mean):
    super(CrossEntropyLoss1d, self).__init__()
    self.reduction = reduction
  
  def forward(self, input, target):
    # softmax
    c = np.mean(input, axis=-1)
    input = np.exp(input - c)
    input = input / np.sum(input, axis=-1, keepdims=True)
    self._input = input  # the value after softmax will be used in back prop.
    
    # nll (1d)
    indices = np.arange(input.shape[0])
    loss_ = -np.log(input[indices, target])
    return np.mean(loss_)
  
  def backward(self, dy):
    _, target = self.inputs
    dy = dy / self._input.shape[0]
    
    input = self._input # CAUTION: self._input is raw data (numpy) whereas target is Tensor
    indices = np.arange(input.shape[0])
    input[indices, target.data] -= 1
    ones_like = np.ones(target.shape)
    
    return detorch.Tensor(input) * dy, detorch.Tensor(ones_like) * dy


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

