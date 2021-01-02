import numpy as np
from ds import Heap

DEFAULT_DTYPE = np.float64


# TODO: topological sort for graph computation
class Variable:
    def __init__(self, data, name='', dtype=DEFAULT_DTYPE, parent_f=None):
        self.data = self.as_array(data, dtype)
        self.name = name
        self.grad = None
        self.parent_f = parent_f
        self.gen = 0 if parent_f is None else parent_f.gen + 1  # generation for graph computation order

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

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

    def backward(self, retain_grads=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        heap = Heap(key=lambda x: -x.gen)
        heap.push(self.parent_f)
        nodes_seen = set()
        while len(heap) > 0:
            f = heap.pop()
            dys = [y().grad for y in f.outputs]
            grads = f.backward(*dys)
            if not isinstance(grads, tuple):
                grads = (grads, )

            for i, grad in zip(f.inputs, grads):
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
        f_nodes = [self.parent_f]
        while f_nodes:
            f = f_nodes.pop()
            for i in f.inputs:
                i.grad = None
                if i.parent_f is not None:
                    f_nodes.append(i.parent_f)


