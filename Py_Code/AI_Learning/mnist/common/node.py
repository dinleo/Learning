import numpy as np


class Reshape:
    def __init__(self):
        self.x_shape = None

    def forward(self, x, shape):
        self.x_shape = x.shape
        return x.reshape(*shape)

    def backward(self, y):
        return y.reshape(self.x_shape)


class Dot:
    def __init__(self):
        self.aT = None
        self.bT = None

    def forward(self, a, b):
        self.aT = a.T  # [128, 100] -> [100, 128]
        self.bT = b.T  # [100, 10] -> [10, 100]
        return np.dot(a, b)  # [128, 100] * [100, 10] -> [128, 10]

    def backward(self, y):
        da = np.dot(y, self.bT)  # [128, 10] * [10, 100] -> [128, 100]
        db = np.dot(self.aT, y)  # [100, 128] * [128, 10] -> [100, 10]
        return da, db   # [100, 10], [128, 100]


class Add:
    def __init__(self):
        self.a_shape = None
        self.b_shape = None

    def forward(self, a, b):
        self.a_shape = a.shape
        self.b_shape = b.shape
        return a + b

    def backward(self, y):
        if self.a_shape != self.b_shape:
            if len(self.a_shape) < len(self.b_shape):
                return np.sum(y, axis=0), y
            else:
                return y, np.sum(y, axis=0)
        else:
            return y, y
