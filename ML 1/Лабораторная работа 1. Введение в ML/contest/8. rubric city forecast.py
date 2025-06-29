from sklearn.base import ClassifierMixin
import numpy as np

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self.pairs = X[['city','modified_rubrics']].value_counts().index.to_numpy()
        self.avg_city_dict = dict()
        for pair in self.pairs:
          city = pair[0]
          rubric_id = pair[1]
          city_mask = X['city'] ==city
          rubric_mask = X['modified_rubrics'] ==rubric_id
          mask = city_mask.to_numpy()*rubric_mask.to_numpy()
          self.avg_city_dict[pair] = np.median(y[mask])

    def predict(self, X=None):
        predict = np.zeros(X.shape[0])
        for pair in self.pairs:
          city = pair[0]
          rubric_id = pair[1]
          city_mask = X['city'] ==city
          rubric_mask = X['modified_rubrics'] ==rubric_id
          mask = city_mask.to_numpy()*rubric_mask.to_numpy()
          predict += mask*self.avg_city_dict[pair]
        return predict