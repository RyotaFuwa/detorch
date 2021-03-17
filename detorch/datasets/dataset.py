from abc import ABC, abstractmethod
import math
import numpy as np


class Dataset(ABC):
  def __init__(self, train=True, transform=None, target_transform=None):
    self.train = train
    self.data = None
    self.label = None
    self.transform = transform if transform is not None else lambda x: x
    self.target_transform = target_transform if target_transform is not None else lambda x: x
    self.prepare()
  
  def __getitem__(self, index):
    assert np.isscalar(index)
    if self.label is None:
      return self.transform(self.data[index]), None
    else:
      return self.transform(self.data[index]), self.target_transform(self.label[index])
    
  def __len__(self):
    return len(self.data)
  
  @abstractmethod
  def prepare(self):
    pass



    
    
