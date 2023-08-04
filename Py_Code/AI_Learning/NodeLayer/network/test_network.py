
import sys
import numpy as np
sys.path.append('C:/Users/dinle/Code/Learning/Py_Code/AI_Learning/mnist')
sys.path.append('C:/Users/dinle/Code/Learning/Py_Code/AI_Learning/NodeLayer/nodes')
from layers import *
from common.layers import Pooling as Pooling2
from common.layers import Convolution as Convolution2
from common.layers import SoftmaxWithLoss as SoftmaxWithLoss2
from common.layers import BatchNormalization as BatchNormalization2
from common.layers import Dropout as Dropout2

w = np.array(range(16)).reshape([2, 2, 2, 2])
x = np.array(range(18)).reshape([1, 2, 3, 3])

b = np.array([1, 2])

x2 = np.array(range(12)).reshape([3, 4])
y = np.array([[0, 0.3, 0.5, 0.2], [0, 0.1, 0.8, 0.1], [0.1, 0, 0.3, 0.6]])
t = np.array([1, 0, 2])

# a1 = Affine(w, b)
# a2 = Affine2(w, b)
# y1 = a1.forward(x)
# y2 = a2.forward(x)
# print(y1)
# print(y2)
# l = (np.array([1] * y1.size)).reshape(y1.shape)
#
# dx1 = a1.backward(l)
# dx2 = a2.backward(l)
# o1 = [dx1, a1.dw, a1.db]
# o2 = [dx2, a2.dW, a2.db]
# print(o1)
# print(o2)
# exit()

# print("X:\n", x)
# print("W:\n", y)
# print("b:\n", b)
# s1 = Convolution(w, b)
# s2 = Convolution2(w, b)
#
# y1 = s1.forward(x)
# y2 = s2.forward(x)
#
# print("forward(Node Architecture):\n", y1)
# print("forward(Text Book):\n", y2)
#
#
# b1 = s1.backward(y1)
# b2 = s2.backward(y2)
#

# print("backward(Node Architecture):\n", b1)
# print("backward(Text Book):\n", b2)

#
# st1 = SoftmaxWithLoss()
# st2 = SoftmaxWithLoss2()
# print("X:\n", x2)
# print("T:\n", t)
# y1 = st1.forward(x2, t)
# y2 = st2.forward(x2, t)
# print("Loss(Node):\n", y1)
# print("Loss(TextBook):\n", y1)
#
# b1 = st1.backward(y1)
# b2 = st2.backward(y2)
# print("dx:\n", b1)
# print("dx:\n", b2)
# #
# # print(l1)
# # print(l2)
# # print(l1 / l2)
# print(y1-t)
#
x = np.array(range(12)).reshape(4, 3)
# x = np.random.randn(12).reshape(4, 3)
N = x.shape[1]
r = np.ones(N)
b = np.ones(N)
st1 = BatchNormalization(r, b)
st2 = BatchNormalization2(r, b)
# print("X:\n", x2)
# print("T:\n", t)
y1 = st1.forward(x)
y2 = st2.forward(x)
# print("(Node):\n", y1)
# print("(TextBook):\n", y2)


b1 = st1.backward(y1)
b2 = st2.backward(y1)
print(b1)
print(b2)
#
s1 = sum(b1)
s2 = sum(b2)
for i in range(4):
    print(b1[i] - b2[i])
# print("Node:\n", b1)
# print("TextBook:\n", b2)
# print(b1/b2)
#
# x = np.array(range(12)).reshape(3, 4)
#
# st1 = Dropout(0.5)
# st2 = Dropout2(0.5)
#
#
# y1 = st1.forward(x)
# y2 = st2.forward(x)
# print("(Node):\n", y1)
# print("(TextBook):\n", y2)
#
#
# b1 = st1.backward(y1)
# b2 = st2.backward(y2)
# print("Node:\n", b1)
# print("TextBook:\n", b2)


# #
# # print(l1)
# # print(l2)
# # print(l1 / l2)
# print(y1-t)
