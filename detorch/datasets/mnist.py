import requests
import os
import gzip
import numpy as np
import detorch
from .dataset import Dataset


class MNIST(Dataset):
  def __init__(self, train=True, transform=None, target_transform=None):
    self.mnist_url = 'http://yann.lecun.com/exdb/mnist/'
    self.train_files = {'target': 'train-images-idx3-ubyte.gz',
                        'label': 'train-labels-idx1-ubyte.gz'}
    self.test_files = {'target': 't10k-images-idx3-ubyte.gz',
                       'label': 't10k-labels-idx1-ubyte.gz'}
    super().__init__(train, transform=transform, target_transform=target_transform)

  def prepare(self):
    files = self.train_files if self.train else self.test_files
    data_path = self._get_file(self.mnist_url + files['target'])
    label_path = self._get_file(self.mnist_url + files['label'])

    self.data = self._load_data(data_path)
    self.label = self._load_label(label_path)
  
  def _get_file(self, url):
    tmp_dir = os.path.join(os.path.expanduser('~'), '.detorch')
    if not os.path.exists(tmp_dir):
      os.makedirs(tmp_dir)
    file_name, _ = os.path.splitext(os.path.basename(url))
    file_dist = os.path.join(tmp_dir, file_name)
    res = requests.get(url)
    if os.path.exists(file_dist):
      return file_dist
    with open(file_dist, 'wb') as fout:
      if res.status_code != 200:
        raise
      fout.write(res.content)
    return file_dist

  def _load_label(self, filepath):
    with gzip.open(filepath, 'rb') as f:
      labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return detorch.Tensor(labels)

  def _load_data(self, filepath):
    with gzip.open(filepath, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return detorch.Tensor(data)
