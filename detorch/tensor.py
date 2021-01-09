from ds import Heap
from .configuration import *
import detorch.functional as F


# TODO: topological sort for graph computation
class Tensor:
    __array_priority__ = 1

    def __init__(self, data, name='', dtype=None, parent_f=None):
        self.data = self.as_array(data, dtype)
        self.name = name
        self.grad = None
        self._parent_f = parent_f
        self.gen = 0 if parent_f is None else parent_f.gen + 1  # generation for graph computation order

    def copy(self):
        r"""copy the tensor. Technically, this operation copies only data and DOES NOT copy parent_f"""
        return Tensor(self.data.copy(), name=self.name, parent_f=self.parent_f)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    @property
    def parent_f(self):
        return self._parent_f

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    # TODO: compute self.grad with Tensor, now self.grad is np.ndarray
    def backward(self, retain_grads=False, create_graph=False):
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))

        heap = Heap(key=lambda x: -x.gen)
        heap.push(self._parent_f)
        nodes_seen = set()
        while len(heap) > 0:
            f = heap.pop()
            dys = [y().grad for y in f.outputs]
            with using_config('enable_backprop', create_graph):
                grads = f.backward(*dys)
                if not isinstance(grads, tuple):
                    grads = (grads, )

                for i, grad in zip(f.inputs, grads):
                    if i.name == 'w1':
                        print(i)
                    if i.grad is not None:
                        i.grad = i.grad + grad  # CAUTION: do not use +=
                    else:
                        i.grad = grad

                    if i.parent_f is not None and i.parent_f not in nodes_seen:
                        nodes_seen.add(i.parent_f)
                        heap.push(i.parent_f)

                if not retain_grads:
                    for y in f.outputs:
                        y().grad = None

    def zero_grad(self):
        self.grad = None
        if self.parent_f is None:
            return
        f_nodes = [self.parent_f]
        while f_nodes:
            f = f_nodes.pop()
            for i in f.inputs:
                i.grad = None
                if i.parent_f is not None:
                    f_nodes.append(i.parent_f)

    @property
    def T(self):
        return F.transpose(self)

    def view(self, *shape):
        return F.view(self, *shape)

    def view_(self, *shape):
        self.data.reshape(shape)
        return self

    def transpose(self):
        return F.transpose(self)

    def transpose_(self):
        self.data = self.data.T
        return self

    def sum(self, axis=None, keepdims=False):
        return F.sum(self, axis=axis, keepdims=keepdims)

    def add(self, other):
        return F.add(self, other)

    def add_(self, other):
        self.data += other.data
        return self

    def sub(self, other):
        return F.sub(self, other)

    def sub_(self, other):
        self.data -= other.data
        return self

    def mul(self, other):
        return F.mul(self, other)

    def mul_(self, other):
        self.data -= other.data
        return self

    def div(self, other):
        return F.div(self, other)

    def div_(self, other):
        self.data -= other.data
        return self

    def __neg__(self):
        return F.neg(self)

    def __add__(self, other):
        return F.add(self, other)

    def __radd__(self, other):
        return F.add(other, self)

    def __sub__(self, other):
        return F.sub(self, other)

    def __rsub__(self, other):
        return F.sub(other, self)

    def __mul__(self, other):
        return F.mul(self, other)

    def __rmul__(self, other):
        return F.mul(other, self)

    def __truediv__(self, other):
        return F.div(self, other)

    def __rtruediv__(self, other):
        return F.div(other, self)

    def __pow__(self, power):
        return F.pow(self, power)

    @staticmethod
    def as_array(data, dtype=None):
        if isinstance(data, np.ndarray):
            if dtype is None:
                return data
            return data.astype(dtype)
        elif np.isscalar(data):
            return np.array(data, dtype=dtype if dtype is not None else Config.default_dtype)
        elif isinstance(data, (tuple, list)):
            return np.array(data, dtype=dtype if dtype is not None else Config.default_dtype)
        else:
            raise TypeError(f"data type of {data} is not compatible. data should be an array-like data structure")

