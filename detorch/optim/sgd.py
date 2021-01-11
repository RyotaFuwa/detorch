import numpy as np
from detorch import Optimizer, no_grad


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.1, momentum=0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum

    @no_grad()
    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

        # Update Using Numpy (faster than Tensor since Tensor wraps numpy object in python atm)
            if self.momentum != 0:
                p_state = self.state[p]
                if 'velocity' not in p_state:
                    p_state['velocity'] = np.zeros_like(p.grad.data)
                v = p_state['velocity']
                v *= self.momentum
                v -= self.lr * p.grad.data
            else:
                v = self.lr * p.grad.data
            p.data += v

        # Update Using Tensor
            #     p_state = self.state[p]
            #     if 'velocity' not in p_state:
            #         p_state['velocity'] = zeros_like(p.grad)
            #     v = p_state['velocity']
            #     v *= self.momentum
            #     v -= self.lr * p.grad
            # else:
            #     v = self.lr * p.grad
            # p += v
