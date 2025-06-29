import numpy as np

from scipy.stats import mode
from sklearn.base import ClassifierMixin

class MostFrequentClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self.y_mode = mode(y).mode

    def predict(self, X=None):
        n = X.shape[0]
        return np.full(n,self.y_mode)