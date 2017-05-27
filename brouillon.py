import numpy as np


a = np.array([[1,2,3], [4,5,6]])
b = np.array([[1,2,3]])
d = np.concatenate((a.ravel(),b.ravel()),axis=0)

c= np.reshape(d[0:6],(2,3))
print(a==c)

print([1]*3)
