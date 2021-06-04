import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import detorch.nn as nn
import detorch.functional as F
import detorch.optim as optim
from detorch.datasets import Dataset, DataLoader


def accuracy(input, target):
  input, target = input.data, target.data
  choices = input.argmax(axis=-1)
  return (choices == target).sum() / len(choices)


class ExampleDataset(Dataset):
  """class to import arff format file and convert it into data with appropriate format to handle."""
  
  def __init__(self, train=True, transform=None, target_transform=None):
    self.train_file = 'data/training.arff'
    self.test_file = 'data/test.arff'
    super().__init__(train, transform=transform, target_transform=target_transform)
  
  def _load_data(self, file):
    arff_data = arff.loadarff(file)
    if isinstance(arff_data, tuple):
      for d in arff_data:
        if isinstance(d, np.ndarray):
          df = pd.DataFrame(d)
          break
      else:
        df = pd.DataFrame(arff_data)
      return df
    else:
      assert True, "not compatible file"
  
  def prepare(self):
    file = self.train_file
    if not self.train:
      file = self.test_file
    df = self._load_data(file)
    self.data, self.label = df[['x', 'y']].to_numpy().astype("float32"), np.where(df['class'] == b'black', 0, 1)
  
  def plot(self, label=None, title=''):
    colors = ['black', 'gray']
    if label is None:
      label = self.label
    label = [colors[i] for i in label]
    plt.scatter(self.data[:, 0], self.data[:, 1], c=label)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    if title == '':
      title = 'Example dataset from CS3423 2nd coursework.'
    plt.title(title)
    plt.show()


class AFFModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(2, 25)
    self.linear2 = nn.Linear(25, 25)
    self.linear3 = nn.Linear(25, 2)
  
  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    return x
  
  """
  def import_config(self, config_filename):
      with open(config_filename, "r") as fin:
          config = yaml.load(fin, Loader=yaml.SafeLoader)
      return config
  """


def main():
  # setting config variables, dataset and model
  epochs = 200
  batch_size = 32
  dataset = ExampleDataset()
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  model = AFFModel()
  criteria = optim.Adam(model.parameters())
  
  # Training the model
  accs = []
  for e in range(epochs):
    for i, x, y in dataloader:
      logits = model(x)
      loss = F.nll_loss(F.softmax(logits), y)
      
      model.zero_grad()
      loss.backward()
      criteria.step()
      
      if i % (dataloader.max_iter * 10) == 0:
        acc = accuracy(logits, y)
        accs.append(acc)
        print("loss: ", loss.data, "acc :", acc)
  
  # Evaluate the model
  testset = ExampleDataset(train=False)
  dataloader = DataLoader(testset, batch_size=batch_size)
  acc = 0
  n_samples = 0
  for i, x, y in dataloader:
    y_hat = model(x)
    choices = y_hat.data.argmax(axis=-1)
    match = choices == y
    plt.scatter(x[~match, 0], x[~match, 1], c=['red' for _ in range((~match).sum())])
    acc += match.sum()
    n_samples += batch_size
  dataset.plot(title='evaluation on test data. (accuracy={})'.format(acc / n_samples))
  plt.show()


if __name__ == '__main__':
  main()
