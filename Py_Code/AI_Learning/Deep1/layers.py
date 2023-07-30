from node import *


class Layer:
    def __init__(self):
        self.get_x = GetValue(None)
        self.dx = None
        self.last_node = None

    def forward(self, x):
        self.get_x.v = x

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dx = self.get_x.dv

        return self.dx


class Affine(Layer):
    def __init__(self, w, b):
        super().__init__()
        self.get_w = GetValue(w)
        self.get_b = GetValue(b)

        self.dw = None
        self.db = None

        self.reshape_node = Reshape(self.get_x, None)
        self.dot_xw_node = Dot(self.reshape_node, self.get_w)
        self.rep_b_node = Repeat(self.get_b, 0, None)
        self.last_node = Add(self.dot_xw_node, self.rep_b_node)

    def forward(self, x):
        self.get_x.v = x
        self.reshape_node.shape = [x.shape[0], -1]
        self.rep_b_node.r = x.shape[0]

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dw = self.get_w.dv
        self.db = self.get_b.dv
        self.dx = self.get_x.dv

        return self.dx


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

        self.neg_node = MulConst(self.get_x, -1)
        self.exp_node = Exp(self.neg_node)
        self.add1_node = AddConst(self.exp_node, 1)
        self.last_node = Reciprocal(self.add1_node)


class Relu(Layer):
    def __init__(self):
        super().__init__()

        self.last_node = Mask(self.get_x, 0)

class Softmax(Layer):
    def __init__(self):
        super().__init__()

        self.norm_node = NormByMax(self.get_x)
        self.exp_node = Exp(self.norm_node)
        self.sum_node = Sum(self.exp_node, 1)
        self.rec_node = Reciprocal(self.sum_node)
        self.rep_node = Repeat(self.rec_node, 1, None)
        self.last_node = Mul(self.exp_node, self.rep_node)

    def forward(self, x):
        self.get_x.v = x
        self.rep_node.r = x.shape[1]

        return self.last_node.forward()


class CrossEntropy:
    def __init__(self):
        self.get_x = GetValue(None)
        self.dx = None
        self.get_t = GetValue(None)

        self.log_node = Log(self.get_x)
        self.mul_node = Mul(self.get_t, self.log_node)
        self.sum_node = Sum(self.mul_node, 1)
        self.neg_node = MulConst(self.sum_node, -1)
        self.last_node = Mean(self.neg_node, 0)

    def forward(self, x, t):
        self.get_x.v = x
        if (x.ndim == 1 and len(t) == 1) or (x.ndim == 2 and t.ndim == 1):
            # not one-hot-vec
            z = np.zeros((t.shape[0], x.shape[x.ndim - 1]))
            for i, v in enumerate(t):
                z[i, v] = 1
            self.get_t.v = z

        else:
            # one-hot-vec
            self.get_t.v = t

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dx = self.get_x.dv

        return self.dx


class SoftmaxWithLoss:
    def __init__(self):
        self.s = Softmax()
        self.c = CrossEntropy()
        self.y = None
        self.t = None

    def forward(self, x, t):
        y = self.s.forward(x)
        o = self.c.forward(y, t)
        self.y = y
        self.t = self.c.get_t.v

        return o

    def backward(self, y):
        dx = self.c.backward(1)
        dx = self.s.backward(dx)

        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.s = Sigmoid()
        self.c = CrossEntropy()

    def forward(self, x, t):
        y = self.s.forward(x)
        o = self.c.forward(y, t)

        return o

    def backward(self, y):
        dx = self.c.backward(1)
        dx = self.s.backward(dx)

        return dx