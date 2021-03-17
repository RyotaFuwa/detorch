import detorch.functional as F
from ds import Heap
from .configuration import *


# TODO: topological sort for graph computation
class Tensor:
  __array_priority__ = 1
  
  def __init__(self, data, name='', dtype=None, grad_fn=None):
    if isinstance(data, Tensor):
      tensor = data
      self.data = tensor.data
    else:
      self.data = self.as_array(data, dtype)
    
    self.name = name
    self.grad = None
    self._grad_fn = grad_fn
    self.gen = 0 if grad_fn is None else grad_fn.gen + 1  # generation for graph computation order
  
  def __len__(self):
    return len(self.data)
  
  def __repr__(self):
    if self.data is None:
      return 'tensor(None)'
    p = str(self.data).replace('\n', '\n' + ' ' * 9)
    grad_fn = f", grad_fn={str(self._grad_fn)}" if self._grad_fn is not None else ''
    return 'tensor(' + p + grad_fn + ')'
  
  # conditional operations don't do backpropagation
  def __le__(self, other):
    other_data = other
    if isinstance(other, Tensor):
      other_data = other.data
    data = self.data <= other_data
    return Tensor(data)
  
  def __lt__(self, other):
    other_data = other
    if isinstance(other, Tensor):
      other_data = other.data
    data = self.data < other_data
    return Tensor(data)
  
  def __ge__(self, other):
    other_data = other
    if isinstance(other, Tensor):
      other_data = other.data
    data = self.data >= other_data
    return Tensor(data)
  
  def __gt__(self, other):
    other_data = other
    if isinstance(other, Tensor):
      other_data = other.data
    data = self.data > other_data
    return Tensor(data)
  
  @property
  def grad_fn(self):
    return self._grad_fn
  
  @property
  def ndim(self):
    return self.data.ndim
  
  @property
  def size(self):
    return self.data.size
  
  @property
  def shape(self):
    return self.data.shape
  
  @property
  def dtype(self):
    return self.data.dtype
  
  def clone(self):
    r"""copy the tensor. Technically, this operation copies only data and DOES NOT copy grad_fn"""
    return Tensor(self.data.copy(), name=self.name, grad_fn=self.grad_fn)
  
  def detach(self):
    self._grad_fn = None
    self.is_leaf = True
    return self
  
  def clear_grad(self):
    self.grad = None
  
  def zero_grad(self): # syntax sugar for clear_grad
    self.grad = None

  @property
  def T(self):
    return F.transpose(self)

  def view(self, *shape):
    return F.view(self, *shape)
  
  def view_(self, *shape):
    self.data = self.data.reshape(shape)
    return self
  
  def flatten(self, preserve_row=False):
    return F.flatten(self, preserve_row=preserve_row)
  
  def flatten_(self, preserve_row=False):
    if preserve_row:
      self.data = self.data.reshape((self.shape[0], -1))
    else:
      self.data = self.data.flatten()
    return self
  
  def squeeze(self):
    return F.squeeze(self)
  
  def squeeze_(self):
    self.data = self.data.squeeze()
    return self

  def unsqueeze(self, dim=-1):
    return F.squeeze(self, dim)

  def unsqueeze_(self, dim=-1):
    self.data = self.data.reshape()
    return self

  def clip(self, min=None, max=None):
    return F.clip(self, min, max)
  
  def clip_(self, min=None, max=None):
    self.data = np.clip(self.data, min, max)
    return self
  
  def transpose(self):
    return F.transpose(self)
  
  def transpose_(self):
    self.data = self.data.T
    return self
  
  def abs(self):
    return F.abs(self)
  
  def abs_(self):
    self.data = self.data.abs()
  
  def sum(self, axis=None, keepdims=False):
    return F.sum(self, axis=axis, keepdims=keepdims)
  
  def add(self, other):
    return F.add(self, other)
  
  def add_(self, other):
    other = self.as_tensor(other)
    self.data += other.data
    return self
  
  def sub(self, other):
    return F.sub(self, other)
  
  def sub_(self, other):
    other = self.as_tensor(other)
    self.data -= other.data
    return self
  
  def mul(self, other):
    return F.mul(self, other)
  
  def mul_(self, other):
    other = self.as_tensor(other)
    self.data *= other.data
    return self
  
  def div(self, other):
    return F.div(self, other)
  
  def div_(self, other):
    other = self.as_tensor(other)
    self.data -= other.data
    return self
  
  def __getitem__(self, slice):
    return F.get_item(self, slice)
  
  def __setitem__(self, key, value):
    self.data[key] = value
    return self
  
  def __neg__(self):
    return F.neg(self)
  
  def __add__(self, other):
    return F.add(self, other)
  
  def __iadd__(self, other):
    return self.add_(other)
  
  def __radd__(self, other):
    return F.add(other, self)
  
  def __sub__(self, other):
    return F.sub(self, other)
  
  def __isub__(self, other):
    return self.sub_(other)
  
  def __rsub__(self, other):
    return F.sub(other, self)
  
  def __mul__(self, other):
    return F.mul(self, other)
  
  def __imul__(self, other):
    return self.mul_(other)
  
  def __rmul__(self, other):
    return F.mul(other, self)
  
  def __truediv__(self, other):
    return F.div(self, other)
  
  def __itruediv__(self, other):
    return self.div_(other)
  
  def __rtruediv__(self, other):
    return F.div(other, self)
  
  def __pow__(self, power):
    return F.pow(self, power)
  
  def __matmul__(self, other):
    return F.mm(self, other)
  
  def __rmatmul__(self, other):
    return F.mm(other, self)

  @staticmethod
  def as_tensor(x, dtype=None):
    if isinstance(x, Tensor):
      return x
    return Tensor(x, dtype=dtype)
  
  @staticmethod
  def as_array(data, dtype=None):
    if isinstance(data, np.ndarray):
      array = data
    elif np.isscalar(data):
      array = np.array(data)
    elif isinstance(data, (tuple, list)):
      array = np.array(data)
    else:
      raise TypeError(f"data type of {data} is not compatible. data should be an array-like data structure")
    if dtype is not None:
      return array.astype(dtype)
    return array

  # TODO: compute self.grad with Tensor, now self.grad is np.ndarray
  def backward(self, retain_grads=False, create_graph=False):
    if self.grad is None:
      self.grad = Tensor(np.ones_like(self.data))

    heap = Heap(key=lambda x: -x.gen)
    heap.push(self._grad_fn)
    nodes_seen = set()
    while len(heap) > 0:
      f = heap.pop()
      dys = [y().grad for y in f.outputs]
      with using_config('enable_backprop', create_graph):
        grads = f.backward(*dys)
        if not isinstance(grads, tuple):
          grads = (grads,)
        for input, grad in zip(f.inputs, grads):
          if input.grad is not None:
            input.grad = input.grad + grad  # CAUTION: do not use +=
          else:
            input.grad = grad
          if input.grad_fn is not None and input.grad_fn not in nodes_seen:
            nodes_seen.add(input.grad_fn)
            heap.push(input.grad_fn)
    
        if not retain_grads:  # If not intended to hold grads, get rid of f.outputs' grads for memory efficiency.
          for y in f.outputs:
            y().grad = None

