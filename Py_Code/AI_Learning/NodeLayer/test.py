import numpy as np
from nodes.axis_node import *
from nodes.one_node import *


class T:
    def __init__(self):
        self.out = None

    def f(self, x):
        y = 2*x
        self.out = y
        return y
org = np.array([1, 2, 3])
tt = T()
n = tt.f(org)
n[0] = 3
print(org)
print(n)
print(tt.out)
