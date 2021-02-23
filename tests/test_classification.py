import math
from abc import ABC, abstractmethod
from typing import Type

import matplotlib.pyplot as plt
import numpy as np


class Generator(ABC):
  CLASSIFIER_TYPE: str
  CLASSES: list = []
  
  @staticmethod
  @abstractmethod
  def generate(X: np.ndarray, *args, **kwargs) -> np.ndarray:
    """

    :param X: np.ndarray with shape of (# of sample, 2) where each sample holds x, y coordinate
    :param args: optional
    :param kwargs: optional
    :return: np.ndarray with shape of (# of sample, 1) where the value represents "a feature" of the sample
    """


class HurricaneGenerator(Generator):
  CLASSIFIER_TYPE = 'median'
  
  @staticmethod
  def generate(X: np.ndarray, twist_coef=2, *args, **kwargs) -> np.ndarray:
    x = X[:, 0]
    y = X[:, 1]
    theta = np.arctan(y / (x + 1e-7)) + np.pi / 2.0
    theta += + np.where(x < 0, np.pi, 0)
    radius = np.sqrt(np.square(x) + np.square(y))
    return (np.power(theta, twist_coef) * radius).reshape((-1, 1))


class CircleGenerator(Generator):
  CLASSIFIER_TYPE = 'median'
  
  @staticmethod
  def generate(X: np.ndarray, steepness=1.0, *args, **kwargs) -> np.ndarray:
    x = X[:, 0]
    y = X[:, 1]
    radius = np.sqrt(np.square(x) + np.square(y))
    return (radius * steepness).reshape((-1, 1))


class CheckerGenerator(Generator):
  """Assume num_of_class is always set to 2"""
  CLASSIFIER_TYPE = 'specified'
  CLASSES = [0.0]
  
  @staticmethod
  def generate(X: np.ndarray, *args, **kwargs) -> np.ndarray:
    num_of_samples = X.shape[0]
    Y = np.zeros((num_of_samples, 1))
    
    for idx in range(num_of_samples):
      if X[idx, 0] > 0:
        if X[idx, 1] > 0:
          Y[idx, 0] = 1
        else:
          Y[idx, 0] = -1
      else:
        if X[idx, 1] > 0:
          Y[idx, 0] = -1
        else:
          Y[idx, 0] = 1
    return Y


"""2D classification data generator"""


def classifier(Y: np.ndarray, type: str = 'median', num_of_class: int = 3, classes=[]):
  """
  :param Y: np.ndarray with shape (# of sample)
  :param type: classifier type, ['median', 'mean', 'specified']
  :param num_of_class:  if type is median or mean, num_of_class will be used.
  :param classes: if type is specified, then this classes parameter will be used.
  :return: np.ndarray with shape (# of sample) filled with int representing class
  """
  if type == 'median':
    if num_of_class is 0:
      assert 'num_of_class can\'t be 0'
    ratios = np.array([100 / num_of_class * i for i in range(1, num_of_class)])
    classes = np.percentile(Y, ratios, axis=0)
  elif type == 'specified':
    if len(classes) == 0:
      assert 'classes is not given'
  else:
    assert 'invalid classifier type'
  
  for idx in range(Y.shape[0]):
    for i, threshold in enumerate(classes):
      if Y[idx, 0] < threshold:
        Y[idx, 0] = i
        break
    else:
      Y[idx, 0] = len(classes)
  return Y.astype('int8')


def generate_data(num_of_class: int, num_of_sample: int, generator: Type[Generator]):
  # limit of num_of_class is set to 100
  if num_of_class > 100:
    num_of_class = 100
  X = np.random.random((num_of_sample, 2)) - 0.5
  X /= np.abs(X).max()
  
  Y = generator.generate(X)
  
  return X, classifier(Y, type=generator.CLASSIFIER_TYPE, num_of_class=num_of_class, classes=generator.CLASSES)


def split_data(x, y, ratio=0.8, shuffle=True):
  indices = np.arange(len(x))
  if shuffle:
    np.random.shuffle(indices)
  split_idx = int(len(x) * ratio)
  return (x[indices[:split_idx]], y[indices[:split_idx]]), (x[indices[split_idx:]], y[indices[split_idx:]])


import detorch.nn as nn
import detorch.optim as optim
import detorch.functional as F


def main():
  # multiple classification
  X, Y = generate_data(3, 5000, HurricaneGenerator)
  (train_x, train_y), (test_x, test_y) = split_data(X, Y, ratio=0.8, shuffle=True)
  plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y)
  plt.show()
  
  epochs = 300
  batch_size = 30
  data_size = len(train_x)
  iters = math.ceil(data_size / batch_size)
  
  model = nn.MLP(input_shape=2, output_shape=3, hidden_layers=[25, 25, 25])
  criterion = optim.SGD(model.parameters(), lr=0.1, momentum=0.85)
  
  for epoch in range(epochs):
    indices = np.random.permutation(data_size)
    sum_loss = 0
    
    for iter in range(iters):
      batch_idx = indices[iter * batch_size: (iter + 1) * batch_size]
      batch_x = train_x[batch_idx]
      batch_y = train_y[batch_idx]
      
      y_pred = model(batch_x)
      loss = F.nnl_loss_1d(F.softmax(y_pred), batch_y)
      
      model.zero_grad()
      loss.backward()
      criterion.step()
      
      sum_loss += loss.data
    print(f"epoch {epoch}, loss: {sum_loss}")


if __name__ == '__main__':
  main()
