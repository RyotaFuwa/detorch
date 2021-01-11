from collections import defaultdict
from abc import ABC


class Optimizer(ABC):
    def __init__(self, parameters):
        if isinstance(parameters, (set, dict)):
            raise TypeError('parameters must be a ordered collection of Tensor')

        self.parameters = list(parameters)
        self.state = defaultdict(dict)
        self.hooks = []

    @staticmethod
    def step(self):
        """"""

    def clear_grad(self):
        for p in self.parameters:
            p.clear_grad()

    # torch syntactical alternative for clear_grad. Technically this replaces p.grad with None, hence clear_grad
    def zero_grad(self):
        self.clear_grad()