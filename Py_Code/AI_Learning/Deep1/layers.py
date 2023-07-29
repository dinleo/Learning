from node import *


class Affine:
    def __init__(self, w, b):
        self.get_w = GetValue(w)
        self.get_b = GetValue(b)
        self.get_x = GetValue(None)

        self.dw = None
        self.db = None
        self.dx = None

        self.reshape_x = Reshape(self.get_x, None)
        self.dot_xw = Dot(self.reshape_x, self.get_w)
        self.last_node = Add(self.dot_xw, self.get_b)

    def forward(self, x):
        self.reshape_x.shape = [x.shape[0], -1]
        self.get_x.v = x

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dw = self.get_w.dv
        self.db = self.get_b.dv
        self.dx = self.get_x.dv

        return self.dx
