import numpy as np
i = 2

a = np.array([[[1,2,3],[4,5,6]] , [[7,8,9],[10,11,12]]])

r = a.shape[i]
s = a.sum(axis=i)
s = np.expand_dims(s, axis=i)
s = np.repeat(s, repeats=r, axis=i)
print(s)

b = np.array([1,2,3])
b2 = np.expand_dims(b, axis=1)
print(np.repeat(b2, repeats= 2, axis=1))

print(np.sum(np.sum(b)))