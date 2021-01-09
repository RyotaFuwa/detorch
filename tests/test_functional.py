import unittest
import numpy as np
import detorch
import detorch.functional as F


class SquareTest(unittest.TestCase):
    def test_gradient_check(self):
        x = detorch.Tensor(np.random.rand(1))
        y = F.square(x)
        y.backward()
        num_grad = numerical_differentiation(square, x, epsilon=1e-4)
        self.assertTrue(np.allclose(x.grad, num_grad))


class ExpTest(unittest.TestCase):
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_differentiation(exp, x, epsilon=1e-4)
        self.assertTrue(np.allclose(x.grad, num_grad))
