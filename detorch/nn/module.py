import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict

from detorch.parameter import Parameter


# TODO: froze weights to do transfer learning
# TODO: refactor
class Module(ABC):
  def __init__(self):
    self._parameters = OrderedDict()
    self._modules = OrderedDict()
  
  def __getattr__(self, key):
    if key in self.__dict__:
      return self.__dict__[key]
    if '_parameters' in self.__dict__:
      _parameters = self.__dict__['_parameters']
      if key in _parameters:
        return self._parameters[key]
    if '_modules' in self.__dict__:
      _modules = self.__dict__['_modules']
      if key in _modules:
        return self._modules[key]
    raise KeyError(key)
  
  def __setattr__(self, key, value):
    if isinstance(value, Parameter):
      self._remove_attr(key)
      self._parameters[key] = value
    elif isinstance(value, Module):
      self._remove_attr(key)
      self._modules[key] = value
    else:
      super().__setattr__(key, value)
  
  def _get_src(self, key):
    if key in self.__dict__:
      return self.__dict__
    if key is self._parameters:
      return self._parameters
    if key is self._modules:
      return self._modules
    return None
  
  def _remove_attr(self, key):
    src = self._get_src(key)
    if src is not None:
      src.pop(key)
  
  def __call__(self, *inputs):
    outputs = self.forward(*inputs)
    if not isinstance(outputs, tuple):
      outputs = (outputs,)
    self.inputs = [weakref.ref(i) for i in inputs]
    self.outputs = [weakref.ref(o) for o in outputs]
    return outputs if len(outputs) > 1 else outputs[0]
  
  @abstractmethod
  def forward(self, *inputs):
    """"""
  
  def apply(self, f):
    f(self)
  
  def parameters(self, recurse=True):
    modules = [('', self)]
    if recurse:
      modules += [(name, module) for name, module in self._modules.items()]
    
    for name, module in modules:
      for k, v in module._parameters.items():
        yield v
  
  def clear_grad(self):
    for param in self.parameters():
      param.clear_grad()
  
  # torch syntactical alternative for clear_grad
  def zero_grad(self):
    self.clear_grad()
  
  def save(self, filepath, save_weights=False):
    pass
  
  @staticmethod
  def load(self, filepath):
    pass
