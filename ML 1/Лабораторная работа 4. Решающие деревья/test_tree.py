import numpy as np
import src_student as src
X = np.array([[1,1,1,1,1,3,3,3,3,3],[10,10,10,50,50,50,10,10,10,10]]).T
y = np.array([1,1,1,0,0,0,0,0,1,1])
# X = np.array([1,10])
# y = np.array([0])
# X_t = np.array([[1],[10]]).T
x = np.arange(2)[None,:]
X = np.array([x]*10)
X = X.reshape(-1,2)
y = np.array([0,1]*5).reshape(-1)
x_t = np.arange(2)[None,:]
y_t = np.array([1])


tree = src.MyDecisionTreeClassifier(max_depth=2)
tree.fit(X, y)
# print(tree.root._right_subtree._node_type)
y_pred = tree.predict(x_t)
print('Predicted:', y_pred)
print('Real:',y_t)
y_proba = tree.predict_proba(x_t)
print('Probabilities:', y_proba)
print(np.apply_along_axis(np.argmax,1,y_proba))

print('----Forest------')
forest = src.MyRandomForestClassifier(n_estimators=100)
forest.fit(X, y)
print(X)
y_pred = forest.predict(x_t)
print('Predicted:', y_pred)
print('Real:',y_t)
y_proba = forest.predict_proba(x_t)
print('Probabilities:', y_proba[0])