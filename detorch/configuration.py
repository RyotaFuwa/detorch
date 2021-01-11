from abc import ABC, abstractmethod
import numpy as np
import functools
import inspect


class Config:
    enable_backprop = True
    default_dtype = np.float64
    init_method = 'xavier'


class _DecoratorContextManager(ABC):
    def __call__(self, f):
        if inspect.isgeneratorfunction(f):
            return self._wrap_generator(f)

        @functools.wraps(f)
        def decorate_context(*args, **kwargs):
            with self.__class__():
                return f(*args, **kwargs)
        return decorate_context

    def _wrap_generator(self, f):
        @functools.wraps(f)
        def decorate_context(*args, **kwargs):
            gen = f(*args, **args)
            while True:
                try:
                    with self.__class__():
                        yield next(gen)
                except StopIteration:
                    break

    @abstractmethod
    def __enter__(self):
        """"""

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """"""


class using_config(_DecoratorContextManager):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __enter__(self):
        self.old_value = getattr(Config, self.key)
        setattr(Config, self.key, self.value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(Config, self.key, self.old_value)


class no_grad(using_config):
    def __init__(self):
        super().__init__('enable_backprop', False)

