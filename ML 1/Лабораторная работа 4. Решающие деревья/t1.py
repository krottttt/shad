import numpy as np

x = np.array([[1, 10], [5, 50]])
print(np.unique(x, return_counts=True))
print(np.apply_over_axes(np.sum, x, [1]))
big_number = 1 << 32
print(big_number)
y = np.array([1,2,3,4,5])
idxes = np.random.randint(0, 5, size=5)
print(idxes)
print(y[idxes])

print(np.arange(10)[None, :])
print(np.arange(10).shape.)
t = np.array([[0]*2]*x.shape[0])
print(t)
t=t/2
print(t)