import math
import numpy as np


class DataLoader:
  def __init__(self, dataset, batch_size, shuffle=True):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.max_iter = math.ceil(len(self.dataset) / batch_size)
    
    self.init_iter()
  
  def init_iter(self):
    self.iter = 0
    if self.shuffle:
      self.indices = np.random.permutation(len(self.dataset))
    else:
      self.indices = np.arange(len(self.dataset))
  
  def __iter__(self):
    return self
  
  def __next__(self):
    if self.iter >= self.max_iter:
      self.init_iter()
      raise StopIteration
  
    i, bs = self.iter, self.batch_size
    indices_mb = self.indices[i * bs: (i + 1) * bs]
    batch = [self.dataset[i] for i in indices_mb]
    self.iter += 1
    return np.array([sample[0] for sample in batch]), np.array([sample[1] for sample in batch])
  
  def next(self):
    return self.__next__()



