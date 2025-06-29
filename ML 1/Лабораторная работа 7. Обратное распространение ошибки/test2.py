import numpy as np

a= np.array([1,2,3])
b = np.array([[1, 2, 3],[6,6,6]])
# print(b-a)
# vec_variance = np.var(b,axis = 0)
# print(vec_variance)
# print(1/vec_variance)
# print(1/6.25)
# print(np.random.binomial(1,0.3))
# mask = 1 - np.random.binomial(3,0.3,size = 3)
# print(mask)
# print(b*mask)
# print(np.ones_like(b))
print(np.pad(b,((1,1),(1,1))))
kernel_size = 3
n = 4
inp = np.random.uniform(-10, 10, size = (1, 2, n, n))
print(inp)
pad_size = kernel_size // 2
out = np.pad(inp,((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size)))
print(out[0,0,:,:])
W = inp = np.random.uniform(-10, 10, size = (2, 2, kernel_size, kernel_size))
print(W[0,0,:,:])