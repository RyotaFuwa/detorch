import unittest

from detorch import *
from detorch.functional import *


def numerical_differentiation(f, x, epsilon=1e-6):
  x0 = Tensor(x.data - epsilon)
  y0 = f(x0)
  x1 = Tensor(x.data + epsilon)
  y1 = f(x1)
  return (y1.data - y0.data) / (2 * epsilon)


class AddTest(unittest.TestCase):
  def test_forward(self):
    a = Tensor(2)
    b = Tensor(3)
    c = add(a, b)
    self.assertTrue(c.data == np.array(5))
  
  def test_gradient(self):
    a = Tensor(np.random.rand(1))
    x = Tensor(np.random.rand(1))
    y = add(a, x)
    y.backward()
    num_grad = numerical_differentiation(lambda x: add(a, x), x, epsilon=1e-4)
    self.assertTrue(np.allclose(x.grad.data, num_grad))


class SubTest(unittest.TestCase):
  def test_forward(self):
    a = Tensor(2)
    b = Tensor(3)
    c = sub(a, b)
    self.assertTrue(c.data == np.array(-1))
  
  def test_gradient(self):
    a = Tensor(np.random.rand(1))
    x = Tensor(np.random.rand(1))
    y = sub(a, x)
    y.backward()
    num_grad = numerical_differentiation(lambda x: sub(a, x), x, epsilon=1e-4)
    self.assertTrue(np.allclose(x.grad.data, num_grad))


class MulTest(unittest.TestCase):
  def test_mul_function(self):
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = mul(a, b)
    self.assertTrue(c.data == np.array(6))
  
  def test_gradient_check(self):
    a = Tensor(np.random.rand(1))
    x = Tensor(np.random.rand(1))
    y = mul(a, x)
    y.backward()
    num_grad = numerical_differentiation(lambda x: mul(a, x), x, epsilon=1e-4)
    self.assertTrue(np.allclose(x.grad.data, num_grad))


class SumTest(unittest.TestCase):
  def test_sum_function(self):
    a = ones(3, 3)
    b = a.sum()
    self.assertTrue(b.data == 9.0)
  
  def test_gradient(self):
    a = ones(3, 3)
    b = a.sum()
    b.backward()
    self.assertTrue(np.alltrue(a.grad.data == a.data))


class SliceTest(unittest.TestCase):
  def test_slice_function(self):
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a[0, :2]
    self.assertTrue(np.alltrue(b.data == np.array([1, 2])))
  
  def test_gradient(self):
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a[0, :2]
    b.backward()
    grad = np.zeros_like(a.data)
    grad[0, :2] = [1, 1]
    self.assertTrue(np.alltrue(a.grad.data == grad))


class ClipTest(unittest.TestCase):
  def test_clip_function(self):
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = F.clip(a, 2, 5)
    self.assertTrue(np.alltrue(b.data == np.array([[2, 2, 3], [4, 5, 5]])))


class GraphTest(unittest.TestCase):
  def test_same_variable_nested(self):
    a = Tensor(3.0)
    b = add(a, a)
    b.backward()
    self.assertTrue(a.grad.data == 2.0)
    
    a.zero_grad()
    b.zero_grad()
    
    b = add(add(a, a), a)
    b.backward()
    self.assertTrue(a.grad.data == 3.0)
  
  def test_branch_out_flow(self):
    a = Tensor(2.0)
    x = square(a)
    x = add(square(x), square(x))
    x.backward()
    self.assertTrue(x.data == 32.0)
    self.assertTrue(a.grad.data == 64.0)
