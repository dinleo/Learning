import numpy as np
from nodes.axis_node import *
from nodes.one_node import *


x1 = np.array([[-0.0688, -0.0688, -0.0688, -0.0688],
 [ 0.3062 , 0.3062,  0.3062,  0.3062],
 [ 0.6812,  0.6812 , 0.6812,  0.6812]])
x2 = np.array([[ 0.375,  0.375,  0.375,  0.375],
 [-0.,    -0.,    -0.,    -0.   ],
 [-0.375, -0.375, -0.375, -0.375]])
print(x1 + x2)