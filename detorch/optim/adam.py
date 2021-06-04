import numpy as np
from detorch import Optimizer, no_grad

EPSILON = 1e-7


class Adam(Optimizer):
  def __init__(self, parameters, lr=0.002, beta1=0.9, beta2=0.999):
    super().__init__(parameters)
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    
    self.beta1_t = 1
    self.beta2_t = 1

  @no_grad()
  def step(self):
    self.beta1_t *= self.beta1
    self.beta2_t *= self.beta2
    for p in self.parameters:
      if p.grad is None:
        continue
      
      # Update Using Numpy (faster than Tensor since Tensor wraps numpy object in python atm)
      p_state = self.state[p]
      if 'm' not in p_state:
        p_state['m'] = np.zeros_like(p.grad.data)
      if 'v' not in p_state:
        p_state['v'] = np.zeros_like(p.grad.data)
      m = p_state['m']
      v = p_state['v']
      p_state['m'] = self.beta1 * m + (1 - self.beta1) * p.grad.data
      p_state['v'] = self.beta2 * v + (1 - self.beta2) * np.square(p.grad.data)
      m_hat = p_state['m'] / (1 - self.beta1_t)
      v_hat = p_state['v'] / (1 - self.beta2_t)
      
      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + EPSILON)
