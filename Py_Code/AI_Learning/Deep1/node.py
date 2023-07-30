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


# OneNode
class OneNode:
    def __init__(self, x_node):
        self.x_node = x_node
        self.out = None


class Exp(OneNode):
    def __init__(self, x_node):
        super().__init__(x_node)

    def forward(self):
        x = self.x_node.forward()
        self.out = np.exp(x)

        return self.out

    def backward(self, y):
        dx = y * self.out
        self.x_node.backward(dx)


class Log(OneNode):
    def __init__(self, x_node):
        super().__init__(x_node)

    def forward(self):
        x = self.x_node.forward()
        self.out = x + 1e-7

        return np.log(self.out)

    def backward(self, y):
        dx = y / self.out
        self.x_node.backward(dx)


# TwoNode
class TwoNode:
    def __init__(self, a_node, b_node):
        self.a_node = a_node
        self.b_node = b_node

    def forward(self):
        pass

    def backward(self, y):
        pass


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


class Mul(TwoNode):
    def __init__(self, a_node, b_node):
        super().__init__(a_node, b_node)
        self.a = None
        self.b = None

    def forward(self):
        self.a = self.a_node.forward()
        self.b = self.b_node.forward()

        return self.a * self.b

    def backward(self, y):
        da = y * self.b
        db = self.a * y

        self.a_node.backward(da)
        self.b_node.backward(db)


class Add(TwoNode):
    def __init__(self, a_node, b_node):
        super().__init__(a_node, b_node)

    def forward(self):
        a = self.a_node.forward()
        b = self.b_node.forward()

        return a + b

    def backward(self, y):
        da = y
        db = y

        self.a_node.backward(da)
        self.b_node.backward(db)


# Structure
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


# Const
class ConstNode:
    def __init__(self, x_node, c):
        self.x_node = x_node
        self.c = c

    def forward(self):
        pass

    def backward(self, y):
        pass


class AddConst(ConstNode):
    def __init__(self, x_node, c=1):
        super().__init__(x_node, c)

    def forward(self):
        x = self.x_node.forward()

        return self.c + x

    def backward(self, y):
        self.x_node.backward(y)


class MulConst(ConstNode):
    def __init__(self, x_node, c=-1):
        super().__init__(x_node, c)
        self.x_node = x_node
        self.c = c

    def forward(self):
        x = self.x_node.forward()

        return x * self.c

    def backward(self, y):
        dx = y * self.c
        self.x_node.backward(dx)


class Reciprocal:
    def __init__(self, x_node):
        self.x_node = x_node
        self.out = None

    def forward(self):
        x = self.x_node.forward()
        self.out = 1 / x

        return self.out

    def backward(self, y):
        dx = (-1) * y
        dx = dx * (self.out * self.out)
        self.x_node.backward(dx)


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


class NormByMax:
    def __init__(self, x_node):
        self.x_node = x_node

    def forward(self):
        x = self.x_node.forward()
        x = x.T
        x = x - np.max(x, axis=0)
        y = x.T

        return y

    def backward(self, y):
        self.x_node.backward(y)


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
