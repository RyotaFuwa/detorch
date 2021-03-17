import numpy as np
import matplotlib.pyplot as plt
from detorch.datasets import MNIST, DataLoader
import detorch.functional as F
import detorch.nn as nn
import detorch.optim as optim


def accuracy(input, target):
  input, target = input.data, target.data
  choices = input.argmax(axis=-1)
  return (choices == target).sum() / len(choices)


if __name__ == '__main__':
  batch_size = 32
  epochs = 5
  
  def transform(x):
    return x.flatten_(preserve_row=True)
  
  dataloader = DataLoader(MNIST(transform=transform), batch_size=batch_size)
  
  model = nn.Sequential([
    nn.Linear(dataloader.data_shape, 1000),
    F.relu,
    nn.Linear(1000, 10),
    F.softmax
  ])
  criteria = optim.SGD(model.parameters())
  
  accs = []
  for e in range(epochs):
    for i, x, y in dataloader:
      y_hat = model(x)
      loss = F.nll_loss(y_hat, y)
      
      model.zero_grad()
      loss.backward()
      criteria.step()
      acc = accuracy(y_hat, y)
      accs.append(acc)
      
      if i % (dataloader.max_iter // 10) == 0:
        print("loss: ", loss.data, "acc :", acc)
  
  plt.plot(accs)
  plt.show()
  
  
  
    
    
    
    
  
  
  
  

