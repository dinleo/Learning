import numpy as np


class Affine2:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class Sigmoid2:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    y1 = y[np.arange(batch_size), t]
    y2 = np.log(y1 + 1e-7)

    return -np.sum(y2) / batch_size


class Softmax2:
    def __init__(self):
        self.y = None  # softmax의 출력

    def forward(self, x):
        self.y = softmax(x)

        return self.y

    def backward(self, y):
        return y * (self.y * (1 - self.y))


class CrossEntropy2:
    def __init__(self):
        self.t = None
        self.x = None

    def forward(self, x, t):
        self.x = x
        if (x.ndim == 1 and len(t) == 1) or (x.ndim == 2 and t.ndim == 1):
            # not one-hot-vec
            z = np.zeros((t.shape[0], x.shape[x.ndim - 1]))
            for i, v in enumerate(t):
                z[i, v] = 1
            self.t = z

        else:
            # one-hot-vec
            self.t = t

        return cross_entropy_error(x, t)

    def backward(self, y):
        self.x = self.x + 1e-7

        return -(self.t/self.x)


class SoftmaxWithLoss2:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
