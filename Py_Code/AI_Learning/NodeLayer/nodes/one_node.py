import numpy as np


class GetValue:
    def __init__(self, v):
        self.v = v
        self.dv = None

    def forward(self):
        # 순전파시 grad 초기화
        self.dv = None
        # 순전파시 value return
        return self.v

    def backward(self, y):
        # 역전파시 흘러들어오는 grad 모두 더함
        if self.dv is None:
            self.dv = y
        else:
            self.dv += y


class OneNode:
    def __init__(self, x_node):
        self.x_node = x_node
        self.out = None


class ConstNode:
    def __init__(self, x_node, c):
        self.x_node = x_node
        self.c = c

    def forward(self):
        pass

    def backward(self, y):
        pass


class Exp(OneNode):
    def __init__(self, x_node):
        super().__init__(x_node)

    def forward(self):
        x = self.x_node.forward()
        self.out = np.exp(x)

        return self.out

    def backward(self, y):
        dx = y * self.out
        # print("Exp backward:\n", dx)
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
        # print("Log backward:\n", dx)
        self.x_node.backward(dx)


class Reciprocal(OneNode):
    def __init__(self, x_node):
        super().__init__(x_node)

    def forward(self):
        x = self.x_node.forward()
        self.out = 1 / x

        return self.out

    def backward(self, y):
        dx = (-1) * y
        dx = dx * (self.out * self.out)
        # print("Reciprocal backward:\n", dx)
        self.x_node.backward(dx)


class AddConst(ConstNode):
    def __init__(self, x_node, c=1):
        super().__init__(x_node, c)

    def forward(self):
        x = self.x_node.forward()

        return self.c + x

    def backward(self, y):
        dx = y
        # print("AddConst backward:\n", dx)
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
        # print("MulConst backward:\n", dx)
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
        dx = y
        # print("NormByMax backward:\n", dx)
        self.x_node.backward(y)
