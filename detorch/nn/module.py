from abc import ABC
from .layer import Layer


class Module(Layer, ABC):
    def summary(self):
        pass

    def save(self, filepath, save_weights=False):
        pass

    @staticmethod
    def load(self, filepath):
        pass

