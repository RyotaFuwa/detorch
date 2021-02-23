import weakref
from abc import ABC, abstractmethod

from detorch import tensor
from .configuration import Config


class Function(ABC):
  _count = -1
  
  # def __init__(self):
  #     self.__class__._count += 1
  
  def __repr__(self):
    # return f"<{self.__class__.__name__}{self.__class__._count}>"
    return f"<{self.__class__.__name__}>"
  
  def __call__(self, *inputs):
    inputs = [tensor.Tensor.as_tensor(i) for i in inputs]
    xs = [i.data for i in inputs]
    
    xs = self.forward(*xs)  # all x in xs is supposed to be np.ndarray
    if not isinstance(xs, tuple):
      xs = (xs,)
    
    grad_fn = self if Config.enable_backprop else None
    self.gen = max([i.gen for i in inputs]) if Config.enable_backprop else 0  # generation for graph
    ys = [tensor.Tensor(x, grad_fn=grad_fn) for x in xs]
    
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
