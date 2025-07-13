import pandas as pd
import scipy
import numpy as np

from src.models.based import BaseEstimatorPerUserCity


class PopularityRecommender(BaseEstimatorPerUserCity):
    def __init__(self, n_items: int = 20):
        self.n_items = n_items
        self.org_popularity = None

    def fit(self, X, y=None):
        """compute popularity of orgs globally and store it in city_to_org_ids"""
        # Вызываем метод родительского класса для инициализации city_to_org_ids
        super().fit(X, y)

        # Вычисляем популярность каждой организации - количество уникальных пользователей
        self.org_popularity = X.groupby('org_id')['user_id'].nunique().sort_values(ascending=False)

        # Ранжируем организации в каждом городе по популярности
        self.city_pop_ranking = {}

        # Для MSK
        msk_orgs = self.city_to_org_ids.msk
        # Создаем Series с индексами org_id и значениями популярности, заполняем 0 для отсутствующих
        msk_popularity = pd.Series(
            index=msk_orgs,
            data=[self.org_popularity.get(org_id, 0) for org_id in msk_orgs]
        ).sort_values(ascending=False)
        self.city_pop_ranking['msk'] = msk_popularity.index.values

        # Для SPB
        spb_orgs = self.city_to_org_ids.spb
        spb_popularity = pd.Series(
            index=spb_orgs,
            data=[self.org_popularity.get(org_id, 0) for org_id in spb_orgs]
        ).sort_values(ascending=False)
        self.city_pop_ranking['spb'] = spb_popularity.index.values

        return self

    def predict_user_org(self, users, orgs):
        """
        predict popular orgs to given users
        :param users: array of user ids
        :param orgs: array of organization ids (from another city)
        :return: array of predictions - top-k most popular orgs for each user
        """
        # Определяем, какой город используем (если пользователи из Москвы, то используем СПб и наоборот)
        city = 'spb' if orgs[0] in self.city_to_org_ids.spb else 'msk'

        # Получаем ранжированный список организаций для соответствующего города
        ranked_orgs = self.city_pop_ranking[city]

        # Обрезаем список до n_items
        top_orgs = ranked_orgs[:self.n_items]

        # Создаем список рекомендаций для каждого пользователя
        # Все пользователи получают одинаковые рекомендации - top-k самых популярных организаций
        recommendations = [top_orgs for _ in range(len(users))]

        return recommendations


class BayesRatingRecommender(PopularityRecommender):
    def __init__(self, n_items: int = 20, alpha=0.1):
        self.alpha = alpha
        self._z = scipy.stats.norm.ppf(1 - alpha / 2)
        self.n_items = n_items

    def get_bayesian_lower_bound_per_group(self, group):
        """
        Calculate the Bayesian lower bound of the Wilson score confidence interval for a group of ratings
        :param group: group of ratings for an organization
        :return: lower bound score
        """
        n = len(group)  # number of ratings
        if n == 0:
            return 0

        # Calculate average rating (p-hat)
        p_hat = group.mean() / 5.0  # normalize to [0, 1]

        # Wilson score interval
        denominator = 1 + (self._z ** 2) / n
        center = (p_hat + (self._z ** 2) / (2 * n)) / denominator
        deviation = self._z * np.sqrt(p_hat * (1 - p_hat) / n + (self._z ** 2) / (4 * n ** 2)) / denominator

        return center - deviation

    def get_bayesian_lower_bound_top(self, orgs: pd.DataFrame):
        """
        Calculate the Bayesian lower bound for all organizations
        :param orgs: DataFrame with organization ratings
        :return: Series with organizations and their scores
        """
        # Group by organization and calculate the lower bound for each
        return orgs.groupby('org_id')['rating'].apply(self.get_bayesian_lower_bound_per_group).sort_values(
            ascending=False)

    def fit(self, X, y=None):
        """
        Fit the model using Bayesian rating approach
        :param X: DataFrame with user-org interactions
        :param y: Not used
        :return: self
        """
        # Вызываем метод родительского класса для инициализации city_to_org_ids
        super(PopularityRecommender, self).fit(X, y)

        # Вычисляем байесовский рейтинг для каждой организации
        self.org_scores = self.get_bayesian_lower_bound_top(X)

        # Ранжируем организации в каждом городе по байесовскому рейтингу
        self.city_score_ranking = {}

        # Для MSK
        msk_orgs = self.city_to_org_ids.msk
        msk_scores = pd.Series(
            index=msk_orgs,
            data=[self.org_scores.get(org_id, 0) for org_id in msk_orgs]
        ).sort_values(ascending=False)
        self.city_score_ranking['msk'] = msk_scores.index.values

        # Для SPB
        spb_orgs = self.city_to_org_ids.spb
        spb_scores = pd.Series(
            index=spb_orgs,
            data=[self.org_scores.get(org_id, 0) for org_id in spb_orgs]
        ).sort_values(ascending=False)
        self.city_score_ranking['spb'] = spb_scores.index.values

        return self

    def predict_user_org(self, users, orgs):
        """
        predict orgs with highest Bayesian rating to users
        :param users: array of user ids
        :param orgs: array of organization ids (from another city)
        :return: array of predictions - top-k highest rated orgs for each user
        """
        # Определяем, какой город используем
        city = 'spb' if orgs[0] in self.city_to_org_ids.spb else 'msk'

        # Получаем ранжированный список организаций для соответствующего города
        ranked_orgs = self.city_score_ranking[city]

        # Обрезаем список до n_items
        top_orgs = ranked_orgs[:self.n_items]

        # Создаем список рекомендаций для каждого пользователя
        recommendations = [top_orgs for _ in range(len(users))]

        return recommendations
