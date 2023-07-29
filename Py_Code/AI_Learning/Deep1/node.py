import numpy as np


class GetValue:
    def __init__(self, v):
        self.v = v
        self.dv = None

    def forward(self):
        return self.v

    def backward(self, y):
        if self.dv is None:
            self.dv = y
        else:
            self.dv += y


class TwoNode:
    def __init__(self, a_node, b_node):
        self.a_node = a_node
        self.b_node = b_node

    def forward(self):
        pass

    def backward(self, y):
        pass


class Reshape:
    def __init__(self, x_node, shape):
        self.x_node = x_node
        self.shape = shape
        self.x_shape = None

    def forward(self):
        x = self.x_node.forward()
        self.x_shape = x.shape

        return x.reshape(*self.shape)

    def backward(self, y):
        y = y.reshape(self.x_shape)
        self.x_node.backward(y)


class Dot(TwoNode):
    def __init__(self, a_node, b_node):
        super().__init__(a_node, b_node)
        self.aT = None
        self.bT = None

    def forward(self):
        a = self.a_node.forward()
        b = self.b_node.forward()
        self.aT = a.T  # [128, 100] -> [100, 128]
        self.bT = b.T  # [100, 10] -> [10, 100]

        return np.dot(a, b)  # [128, 100] * [100, 10] -> [128, 10]

    def backward(self, y):
        da = np.dot(y, self.bT)  # [128, 10] * [10, 100] -> [128, 100]
        db = np.dot(self.aT, y)  # [100, 128] * [128, 10] -> [100, 10]
        self.a_node.backward(da)
        self.b_node.backward(db)


class Add(TwoNode):
    def __init__(self, a_node, b_node):
        super().__init__(a_node, b_node)
        self.a_shape = None
        self.b_shape = None

    def forward(self):
        a = self.a_node.forward()
        b = self.b_node.forward()
        self.a_shape = a.shape
        self.b_shape = b.shape

        return a + b

    def backward(self, y):
        if self.a_shape != self.b_shape:
            if len(self.a_shape) < len(self.b_shape):
                da = np.sum(y, axis=0)
                db = y
            else:
                da = y
                db = np.sum(y, axis=0)
        else:
            da = y
            db = y

        self.a_node.backward(da)
        self.b_node.backward(db)
