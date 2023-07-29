from layers import *

w = np.array([1]*12).reshape(4,3)
x = np.array([[1,1,1,1], [2,2,2,2]])
b = np.array([1,1,1])


a = Affine(w, b)

y = a.forward(x)
print(y)
l = (np.array([1] * y.size)).reshape(y.shape)
print(l)
dx = a.backward(l)
dw = a.dw
db = a.db
print(dx, dw, db)