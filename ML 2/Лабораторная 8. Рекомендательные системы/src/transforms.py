from sklearn.base import TransformerMixin

from src.models.based import BaseEstimator

import pandas as pd


class UserInfoJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, user_info):
        self._user_info = user_info.set_index('user_id').rename(columns=lambda s: 'user_' + s)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        join user info (city is important) to reviews dataframe
        :param X:
        :return:
        """
        X = pd.merge(X,self._user_info,on='user_id',how='left')
        # return pd.merge(X,self._user_info,on='user_id',how='left')
        pass

class OrgInfoJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, org_info):
        self._org_info = org_info.set_index('org_id').rename(columns=lambda s: 'org_' + s)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        join org info (city is important) to reviews dataframe, mind that orgs are not always present in input dataframe
        say, in case of prediction
        :param X:
        :return:
        """
        X = pd.merge(X,self._user_info,on='org_id',how='left')
        # return pd.merge(X,self._user_info,on='org_id',how='left')
        pass

class MissingRatingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, value=None):
        self.strategy = strategy
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        impute missing rating values (there are some in original data)
        :param X:
        :return:
        """
        if self.strategy == 'mean imputer':
            X['rating'].fillna(X['rating'].mean(), inplace=True)
        elif self.strategy == 'constant imputer':
            X['rating'].fillna(self.value, inplace=True)
        else:
            raise NotImplementedError('strategy must be "mean imputer" or "constant imputer"')