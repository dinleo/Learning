import numpy as np
from nodes.axis_node import *
from nodes.one_node import *


a = np.random.rand(12).reshape([4,3])
print("a", a)

g = GetValue(a)
m = Max(g, axis=1)

y = m.forward()
print("y", y)
m.backward(y)
print("dx", g.dv)