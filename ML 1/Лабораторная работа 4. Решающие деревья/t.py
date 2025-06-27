import numpy as  np
from scipy.stats import mode
y_l = np.array([[2,3,4,1,6],[80,100,20,30,50]])
r_l = np.array([1,0,0,1,0])
idx = np.where(y_l[0]<=3)
print(r_l[idx])
print(y_l[:,idx])
print(mode(r_l)[0])
print(np.unique(r_l, return_counts=True)[1]/len(r_l))
print(y_l.T[idx])
print(np.apply_over_axes(np.sum, y_l.T, [1]))
print(np.apply_along_axis(np.sum,1, y_l.T))
print(r_l[:-1])

x = np.array([[3, 10], [3, 10], [3, 10], [3, 10]])
print(len(np.unique(x,axis=0)))
# data_3d = np.random.randint(1, 10, (3, 4, 5))
#
# # Применяем сумму по осям 0 и 1
# result = np.apply_over_axes(np.sum, data_3d, [0, 1])
# print("\nФорма исходного массива:", data_3d.shape)
# print("Форма результата после суммирования по осям 0 и 1:", result.shape)
# print("Результат (упрощенный):", result.ravel())