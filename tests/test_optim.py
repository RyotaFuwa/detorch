import detorch.functional as F
from detorch import Tensor

x0 = Tensor(0)
x1 = Tensor(2)


def vanilla_gradient_decent(f, *xs, lr=0.001, iters=5000):
  for i in range(iters):
    y = f(*xs)
    
    if i % 100 == 0:
      print("--- processing ({:.1f}%) ---".format(i / iters * 100))
      for i, x in enumerate(xs):
        print("x{}: {:.4f}".format(i, x.data))
      print("y: {:.4f}".format(y.data))
    
    y.zero_grad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad


if __name__ == '__main__':
  x0 = Tensor(0)
  x1 = Tensor(2)
  f = F.rosenbrock
  vanilla_gradient_decent(f, x0, x1)
