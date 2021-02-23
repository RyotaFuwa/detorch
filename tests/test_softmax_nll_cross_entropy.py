import numpy as np
import detorch
import detorch.functional as F

if __name__ == '__main__':
  x = detorch.Tensor(np.random.randn(100, 10))
  t = np.random.randint(10, size=(100,))
  y_softmax_nll = F.nll_loss(F.softmax(x), t)
  y_cross_entropy = F.cross_entropy_loss(x, t)
  print(y_softmax_nll, y_cross_entropy)
  
  for i in range(10):
    y_softmax_nll.backward()
    grad_softmax_nll = x.grad
    x.clear_grad()
    y_cross_entropy.backward()
    grad_cross_entropy = x.grad
    x.clear_grad()

  print(np.allclose(y_softmax_nll.data, y_cross_entropy.data))
  print(np.allclose(grad_softmax_nll.data, grad_cross_entropy.data))

