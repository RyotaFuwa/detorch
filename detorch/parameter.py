import numpy as np
from detorch import Config, Tensor


class Parameter(Tensor):
    def __repr__(self):
        if self.data is None:
            return 'parameter(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'parameter(' + p + ')'

