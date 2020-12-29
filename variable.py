

class Variable:
    def __init__(self, data, parent=None):
        self.data = data
        self.grad = None
        self._parent = parent

    def set_parent(self, parent):
        self._parent = parent

    def backward(self):
        f_nodes = [self._parent]
        while f_nodes:
            f = f_nodes.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x._parent is not None:
                f_nodes.append(x._parent)


