from detorch.functional import relu
from detorch.nn import Module
from .linear import Linear


# test module
class TwoLayerNet(Module):
  def __init__(self):
    super().__init__()


class MLP(Module):
  def __init__(self, input_shape, output_shape, hidden_layers=[], activation=relu):
    super().__init__()
    self.activation = activation
    self.layers = []
    if len(hidden_layers) > 0:
      self.layers.append(Linear(input_shape, hidden_layers[0]))
      for i in range(1, len(hidden_layers)):
        layer = Linear(hidden_layers[i - 1], hidden_layers[i])
        self.layers.append(layer)
      self.layers.append(Linear(hidden_layers[-1], output_shape))
    else:
      self.layers = [Linear(input_shape, output_shape)]
  
  def forward(self, x):
    for linear in self.layers:
      x = self.activation(linear(x))
    return x
