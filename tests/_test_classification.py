import torch
import torch.optim as optimt
import torch.nn as nnt


def main_t():
  class FNN3(nnt.Module):
    def __init__(self):
      super().__init__()
      self.layer1 = nnt.Linear(2, 25)
      self.layer2 = nnt.Linear(25, 25)
      self.layer3 = nnt.Linear(25, 3)
    
    def forward(self, x):
      x = self.layer1(x)
      x = torch.relu(x)
      x = self.layer2(x)
      x = torch.relu(x)
      x = self.layer3(x)
      return x
  
  # multiple classification by pytorch
  X, Y = generate_data(3, 5000, HurricaneGenerator)
  (train_x, train_y), (test_x, test_y) = split_data(X, Y, ratio=0.8, shuffle=True)
  train_x, train_y = torch.tensor(train_x).float(), torch.tensor(train_y).squeeze()
  test_x, test_y = torch.tensor(test_x).float(), torch.tensor(test_y).squeeze()
  
  epochs = 300
  batch_size = 30
  data_size = len(train_x)
  iters = math.ceil(data_size / batch_size)
  
  model = FNN3()
  criterion = optimt.SGD(model.parameters(), lr=0.01, momentum=0.95)
  loss = nnt.CrossEntropyLoss()
  
  for epoch in range(epochs):
    indices = np.random.permutation(data_size)
    sum_loss = 0
    
    for iter in range(iters):
      batch_idx = indices[iter * batch_size: (iter + 1) * batch_size]
      batch_x = train_x[batch_idx]
      batch_y = train_y[batch_idx]
      
      batch_y_pred = model(batch_x)
      loss_ = loss(batch_y_pred, batch_y.long())
      
      model.zero_grad()
      loss_.backward()
      criterion.step()
      
      sum_loss += loss_.data
    print(f"epoch {epoch}, loss: {sum_loss}")
  
  test_y_pred = model(test_x)
  test_y_pred = test_y_pred.max(axis=1)
  plt.scatter(test_x_[:, 0], test_x_[:, 1], c=test_y_pred)
  plt.show()
