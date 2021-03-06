import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as tF

import detorch.functional as F
import detorch.nn as nn
from detorch.optim import SGD


def main_detorch(x, y):
  lr = 0.2
  iters = 10000
  # model = nn.TwoLayerNet(1, 1, 10)
  model = nn.Sequential([
    nn.Linear(1, 10),
    F.sigmoid,
    nn.Linear(10, 1)
  ])
  criterion = SGD(model.parameters(), lr=0.1, momentum=0.9)
  
  for i in range(iters):
    y_hat = model(x)
    loss = F.mse_loss(y, y_hat)
    
    model.clear_grad()
    loss.backward()
    
    criterion.step()
    
    if i % 1000 == 0:
      print(loss)
  
  return model(x).data


def main_torch(x, y):
  x = torch.Tensor(x).unsqueeze(1)
  y = torch.Tensor(y).unsqueeze(1)
  
  class FNN2(tnn.Module):
    def __init__(self):
      super().__init__()
      self.layer1 = tnn.Linear(1, 10)
      self.layer2 = tnn.Linear(10, 1)
    
    def forward(self, x):
      x = self.layer1(x)
      x = torch.sigmoid(x)
      x = self.layer2(x)
      return x
  
  lr = 0.2
  iters = 10000
  model = FNN2()
  for i in range(iters):
    y_hat = model(x)
    loss = tF.mse_loss(y, y_hat)
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
      for p in model.parameters(): p -= p.grad * lr
    
    if i % 1000 == 0:
      print(loss)
  
  return model(x).squeeze().detach().numpy()


if __name__ == '__main__':
  np.random.seed(0)
  x = np.random.rand(100, 1)
  y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
  
  start = time.time()
  y_hat_detorch = main_detorch(x, y)
  time_detorch = time.time() - start
  y_hat_torch = main_torch(x, y)
  time_torch = time.time() - (start + time_detorch)
  
  print(time_detorch, time_torch)
  
  plt.plot(x, y, 'bo')
  plt.plot(x, y_hat_detorch, 'go')
  plt.plot(x, y_hat_torch, 'yo')
  plt.show()
