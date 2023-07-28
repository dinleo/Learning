from node import *


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.dW = None
        self.db = None

        self.nodes = {
            'reshape': Reshape(),
            'dot': Dot(),
            'add': Add()
        }

    def forward(self, x):
        y = self.nodes['reshape'].forward(x, [x.shape[0], -1])
        y = self.nodes['dot'].forward(y, self.W)
        y = self.nodes['add'].forward(y, self.b)

        return y

    def backward(self, y):
        dx, self.db = self.nodes['add'].backward(y)
        dx, self.dW = self.nodes['dot'].backward(dx)
        dx = self.nodes['reshape'].backward(dx)

        return dx
