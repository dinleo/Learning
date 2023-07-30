from layers import *
import sys
sys.path.append('C:/Users/dinle/Code/Learning/Py_Code/AI_Learning/mnist')
from common.layers import Pooling as Pooling2

w = np.array(range(36)).reshape([2, 2, 3, 3])
x = np.array(range(18)).reshape([1, 2, 3, 3])

b = np.array([1, 2])


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

s1 = Pooling(2, 2)
s2 = Pooling2(2, 2)

y1 = s1.forward(x)
y2 = s2.forward(x)

print(y1)
print(y2)


b1 = s1.backward(y1)
b2 = s2.backward(y2)
#

print(b1)
print(b2)

#
# st1 = SoftmaxWithLoss()
# st2 = SoftmaxWithLoss2()
#
# y1 = st1.forward(x, t)
# y2 = st2.forward(x, t)
# print(y1)
# l1 = st1.backward(y1)
# l2 = st2.backward(y2)
#
# print(l1)
# print(l2)
# print(l1 / l2)
