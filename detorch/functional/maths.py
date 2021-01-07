import numpy as np
import detorch


class Exp(detorch.Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dy):
        x = self.inputs
        return np.exp(x) * dy


class Log(detorch.Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, dy):
        x = self.inputs[0].data
        return dy / x


class Sin(detorch.Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, dy):
        x, = self.inputs
        return cos(x) * dy


class Cos(detorch.Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, dy):
        x, = self.inputs
        return -sin(x) * dy


class Tan(detorch.Function):
    def forward(self, x):
        return np.tan(x)

    def backward(self, dy):
        x = self.inputs
        return dy / cos(x) ** 2


class Tanh(detorch.Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, dy):
        y = self.outputs[0]()
        return dy * (1 - y ** 2)


def exp(input):
    return Exp()(input)


def log(input):
    return Log()(input)


def sin(input):
    return Sin()(input)


def cos(input):
    return Cos()(input)


def tan(input):
    return Tan()(input)


def tanh(input):
    return Tanh()(input)


def square(input):
    return detorch.pow(input, 2)


def cube(input):
    return detorch.pow(input, 3)


def sqrt(input):
    return detorch.pow(input, 0.5)


def sphere(x, y):
    return x ** 2 + y ** 2


def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
    return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
           (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))


def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2


def tmp_test():
    x = detorch.Tensor(1)
    y = detorch.Tensor(1)
    z = sphere(x, y)
    z.backward()
    print(x.grad, y.grad)
    z.zero_grad()

    z = matyas(x, y)
    z.backward()
    print(x.grad, y.grad)
    z.zero_grad()

    z = goldstein(x, y)
    z.backward()
    print(x.grad, y.grad)


if __name__ == '__main__':
    tmp_test()

