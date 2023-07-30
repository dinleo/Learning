from nodes.one_node import *
from nodes.two_node import *
from nodes.structure_node import *
from nodes.axis_node import *


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

        self.dW = None
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
        self.dW = self.get_w.dv
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


class Convolution(Layer):
    def __init__(self, w, b, stride=1, pad=0):
        super().__init__()
        self.get_w = GetValue(w)
        self.get_b = GetValue(b)

        self.dW = None
        self.db = None
        self.stride = stride
        self.pad = pad

        self.im2col_node = Img2Matrix(self.get_x)
        self.w_reshape_node = Reshape(self.get_w, None)
        self.w_t_node = T(self.w_reshape_node)
        self.dot_node = Dot(self.im2col_node, self.w_t_node)
        self.rep_b_node = Repeat(self.get_b, 0, None)
        self.add_node = Add(self.dot_node, self.rep_b_node)
        self.y_reshape_node = Reshape(self.add_node, None)
        self.last_node = Transpose(self.y_reshape_node, None)

    def forward(self, x):
        FN, C, FH, FW = self.get_w.v.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        filter_shape = {
            'fh': FH,
            'fw': FW,
            'stride': self.stride,
            'pad': self.pad
        }

        self.get_x.v = x
        self.im2col_node.filter_shape = filter_shape
        self.w_reshape_node.shape = [FN, -1]
        self.rep_b_node.r = N * out_h * out_w
        self.y_reshape_node.shape = [N, out_h, out_w, -1]
        self.last_node.shape = [0, 3, 1, 2]

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dW = self.get_w.dv
        self.db = self.get_b.dv
        self.dx = self.get_x.dv

        return self.dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx