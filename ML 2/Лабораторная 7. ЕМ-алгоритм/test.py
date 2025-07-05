import numpy as np

s = 0.2
np.random.seed(42)
print(s+np.random.rand()/(1/(1-2*s)))
