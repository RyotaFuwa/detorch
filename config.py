import contextlib

import numpy as np


class Config:
  enable_backprop = True
  default_dtype = np.float64


@contextlib.contextmanager
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)


@contextlib.contextmanager
def no_grad():
  return using_config('enable_backprop', False)
