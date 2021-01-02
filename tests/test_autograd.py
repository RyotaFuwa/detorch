import unittest
import numpy as np
from variable import Variable
from function import *


def numerical_differentiation(f, x, epsilon):
    x0 = Variable(x.data - epsilon)
    y0 = f(x0)
    x1 = Variable(x.data + epsilon)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * epsilon)


class SquareTest(unittest.TestCase):
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
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


class AddTest(unittest.TestCase):
    def test_add_function(self):
        a = Variable(2)
        b = Variable(3)
        c = add(a, b)
        self.assertTrue(c.data == np.array(5))

    def test_gradient_check(self):
        a = Variable(np.random.rand(1))
        x = Variable(np.random.rand(1))
        y = add(a, x)
        y.backward()
        num_grad = numerical_differentiation(lambda x: add(a, x), x, epsilon=1e-4)
        self.assertTrue(np.allclose(x.grad, num_grad))


class MulTest(unittest.TestCase):
    def test_mul_function(self):
        a = Variable(2.0)
        b = Variable(3.0)
        c = mul(a, b)
        self.assertTrue(c.data == np.array(6))

    def test_gradient_check(self):
        a = Variable(np.random.rand(1))
        x = Variable(np.random.rand(1))
        y = mul(a, x)
        y.backward()
        num_grad = numerical_differentiation(lambda x: mul(a, x), x, epsilon=1e-4)
        self.assertTrue(np.allclose(x.grad, num_grad))


class GraphTest(unittest.TestCase):
    def test_same_variable_nested(self):
        a = Variable(3.0)
        b = add(a, a)
        b.backward()
        self.assertTrue(a.grad == 2.0)

        b.zero_grad()

        b = add(add(a, a), a)
        b.backward()
        self.assertTrue(a.grad == 3.0)

    def test_branch_out_flow(self):
        a = Variable(2.0)
        x = square(a)
        x = add(square(x), square(x))
        x.backward()
        self.assertTrue(x.data == 32.0)
        self.assertTrue(a.grad == 64.0)






