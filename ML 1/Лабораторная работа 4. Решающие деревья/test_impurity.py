from src_student import gini,weighted_impurity
import numpy as np

y_l = np.array([1,0,1])
y_r = np.array([0,1,0,1,1])
print(gini(y_l))
print(gini(y_r))
print(np.concatenate([y_l,y_r],axis=0))
print(weighted_impurity(y_l,y_r))