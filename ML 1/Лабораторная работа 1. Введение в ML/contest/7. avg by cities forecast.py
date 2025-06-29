from sklearn.base import RegressorMixin
import numpy as np

class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        self.cities = X['city'].unique()
        self.avg_city_dict = dict()
        for city in self.cities:
          mask = X['city'] == city
          self.avg_city_dict[city] = np.mean(y[mask.to_numpy()])

    def predict(self, X=None):
        predict = np.zeros(X.shape[0])
        for city in self.cities:
          mask = X['city'] == city
          predict += mask.to_numpy()*self.avg_city_dict[city]
        return predict