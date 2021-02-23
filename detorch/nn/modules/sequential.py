from collections import Counter
from detorch.functional import relu, sigmoid
import detorch.nn as nn


# test module
class TwoLayerNet(nn.Module):
  def __init__(self, input_weight, output_weight, hidden_weight):
    super().__init__()
    self.layer1 = nn.Linear(input_weight, hidden_weight)
    self.layer2 = nn.Linear(hidden_weight, output_weight)

  def forward(self, x):
    x = self.layer1(x)
    x = sigmoid(x)
    x = self.layer2(x)
    return x


class Sequential(nn.Module):
  def __init__(self, layers):
    super().__init__()
    self._layers = []
    self.key_counts = Counter()
    for layer in layers:
      if not callable(layer):
        raise ValueError("All layers have to be callable on tensors such as Layer(tensors: Tensor, ...) -> Tensor, ...")
      if isinstance(layer, nn.Module):
        key = '_' + layer.__class__.__name__
      else:
        key = "_" + layer.__name__
      self.key_counts[key] += 1
      key = key + str(self.key_counts[key] - 1)
      self.__setattr__(key, layer)
      self._layers.append(layer)
  
  def forward(self, x):
    for layer in self._layers:
      x = layer(x)
    return x
