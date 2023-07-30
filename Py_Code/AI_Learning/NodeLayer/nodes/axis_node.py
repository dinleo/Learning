import numpy as np


class Sum:
    def __init__(self, x_node, axis=0):
        self.x_node = x_node
        self.axis = axis
        self.r = None
        self.shape = None

    def forward(self):
        x = self.x_node.forward()

        self.r = x.shape[self.axis]
        x = np.sum(x, axis=self.axis)

        return x

    def backward(self, y):
        dx = np.expand_dims(y, axis=self.axis)
        dx = np.repeat(dx, self.r, axis=self.axis)

        self.x_node.backward(dx)


class Repeat:
    def __init__(self, x_node, axis, r):
        self.x_node = x_node
        self.axis = axis
        self.r = r

    def forward(self):
        x = self.x_node.forward()
        y = np.expand_dims(x, axis=self.axis)
        y = np.repeat(y, repeats=self.r, axis=self.axis)

        return y

    def backward(self, y):
        dx = np.sum(y, axis=self.axis)
        self.x_node.backward(dx)


class Mean:
    def __init__(self, x_node, axis=0):
        self.x_node = x_node
        self.axis = axis
        self.r = None

    def forward(self):
        x = self.x_node.forward()
        self.r = x.shape[self.axis]
        x = np.sum(x, axis=self.axis) / self.r

        return x

    def backward(self, y):
        dx = np.expand_dims(y, axis=self.axis)
        dx = np.repeat(dx, self.r, axis=self.axis) / self.r

        self.x_node.backward(dx)


class Mask:
    def __init__(self, x_node, t=0):
        self.x_node = x_node
        self.t = t
        self.mask = None

    def forward(self):
        x = self.x_node.forward()
        self.mask = (x <= self.t)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, y):
        y[self.mask] = 0
        dx = y
        self.x_node.backward(dx)