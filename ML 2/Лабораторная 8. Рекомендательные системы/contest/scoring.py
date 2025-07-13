import secrets
from collections import defaultdict
from copy import copy

import numpy as np
import pandas as pd

from src.cross_val import DaysTimeSeriesSplit
from src.data_loading import Data


def rel_per_user(series, k=20):
    """
    computes relevance per user
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :param k: how many predictions to consider
    :return: np.ndarray(dtype=bool) with encoded relevance
    """
    # Получаем массив предсказаний и таргетов
    predictions = series['predictions']
    targets = series['targets']

    # Обрезаем предсказания до k элементов
    if len(predictions) > k:
        predictions = predictions[:k]

    # Вычисляем релевантность каждого предсказания
    relevance = np.array([pred in targets for pred in predictions], dtype=bool)

    return relevance


def mnap_at_k_per_user(series, k=20):
    """
    computes MNAP per user (use relevance), should output `nan` if no interactions for given user are in target
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :param k: how many predictions to consider
    :return: MNAP score (float [0..1])
    """
    # Получаем релевантность для каждого предсказания
    relevance = series['rel']
    targets = series['targets']

    # Если нет релевантных объектов для пользователя, возвращаем nan
    if len(targets) == 0:
        return np.nan

    # Вычисляем знаменатель - минимум из количества взаимодействий пользователя и k
    denominator = min(len(targets), k)

    # Если ни одно предсказание не было сделано, возвращаем 0
    if len(relevance) == 0:
        return 0.0

    # Обрезаем релевантность до k элементов
    if len(relevance) > k:
        relevance = relevance[:k]

    # Вычисляем precision@i для каждого i
    precision_at_i = np.cumsum(relevance) / np.arange(1, len(relevance) + 1)

    # Вычисляем NAP@k
    nap_at_k = np.sum(relevance * precision_at_i) / denominator

    return nap_at_k


def hit_rate_per_user(series):
    """
    computes hit rate per user (use relevance), should output `nan` if no interactions for given user are in target
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :return: hit rate score (float [0..1])
    """
    # Получаем релевантность для каждого предсказания и таргеты
    relevance = series['rel']
    targets = series['targets']

    # Если нет релевантных объектов для пользователя, возвращаем nan
    if len(targets) == 0:
        return np.nan

    # Проверяем, есть ли хотя бы одно релевантное предсказание
    hit_rate = 1.0 if np.any(relevance) else 0.0

    return hit_rate


def mrr_per_user(series):
    """
    computes MRR per user (use relevance), should output `nan` if no interactions for given user are in target
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :return: mrr score (float [0..1])
    """
    # Получаем релевантность для каждого предсказания и таргеты
    relevance = series['rel']
    targets = series['targets']

    # Если нет релевантных объектов для пользователя, возвращаем nan
    if len(targets) == 0:
        return np.nan

    # Если ни одно предсказание не было релевантным, возвращаем 0
    if not np.any(relevance):
        return 0.0

    # Находим позицию первого релевантного предсказания (индексация с 1)
    first_relevant_pos = np.argmax(relevance) + 1

    # Вычисляем reciprocal rank
    reciprocal_rank = 1.0 / first_relevant_pos

    return reciprocal_rank


def calc_coverage(exploded_predict, orgs):
    """
    computes coverage over entire prediction series
    :param exploded_predict: exploded (one prediction per row) prediction series
    :param orgs: dataframe with org information
    :return: coverage score (float [0..1])
    """
    # Получаем количество уникальных рекомендованных организаций
    unique_recommended_orgs = exploded_predict.unique()
    unique_recommended_count = len(unique_recommended_orgs)

    # Получаем общее количество уникальных организаций в тренировочных данных
    total_unique_orgs = len(orgs)

    # Вычисляем coverage
    coverage = unique_recommended_count / total_unique_orgs

    return coverage


def calc_surprisal(exploded_predict, x_train):
    """
    computes surprisal over entire dataset
    :param exploded_predict: exploded (one prediction per row) prediction series
    :param x_train: interactions (reviews) dataframe
    :return: surprisal score (float [0..1])
    """
    # Получаем общее количество взаимодействий
    total_interactions = len(x_train)

    # Считаем количество взаимодействий для каждой организации
    org_interactions = x_train['org_id'].value_counts()

    # Для каждой рекомендованной организации вычисляем self-information
    # Формула: -log2(max(|interactions with item|, 1) / |all interactions|)

    # Для каждой рекомендации находим количество взаимодействий
    recommended_orgs_interactions = exploded_predict.map(
        lambda org_id: org_interactions.get(org_id, 0)
    )

    # Вычисляем self-information для каждой рекомендации
    self_information = -np.log2(np.maximum(recommended_orgs_interactions, 1) / total_interactions)

    # Группируем по индексам (пользователям) и вычисляем среднее self-information для каждого пользователя
    user_surprisal = exploded_predict.groupby(level=0).apply(
        lambda user_recs: self_information.loc[user_recs.index].mean()
    )

    # Вычисляем общий surprisal как среднее по всем пользователям
    surprisal = user_surprisal.mean() / np.log2(total_interactions)

    return surprisal


class Scorer:
    def __init__(self, k: int, cv_splitter: DaysTimeSeriesSplit, data: Data):
        self.k = k
        self._cv_splitter = cv_splitter
        self._data = data
        self._score_table = pd.DataFrame()

    def leaderboard(self):
        return self._score_table

    def scoring_fn(self, predict, x_train, x_test):
        predict = predict.rename('predictions')
        x_test_with_cities = (x_test
                              .join(self._data.users.set_index('user_id')['city'], on='user_id')
                              .join(self._data.organisations.set_index('org_id')['city'], on='org_id',
                                    rsuffix='_user',
                                    lsuffix='_org'))
        list_compressed_target = x_test_with_cities.groupby('user_id').apply(
            lambda s: np.array(s['org_id'][(s['rating'] >= 4.0) & (s['city_user'] != s['city_org'])]),
            include_groups=False).rename('targets')
        joined = pd.merge(predict, list_compressed_target, left_index=True, right_index=True)
        joined['rel'] = joined.apply(rel_per_user, k=self.k, axis=1)
        joined['mnap'] = joined.apply(mnap_at_k_per_user, k=self.k, axis=1)
        joined['hit_rate'] = joined.apply(hit_rate_per_user, axis=1)
        joined['mrr'] = joined.apply(mrr_per_user, axis=1)
        metric_vals = joined[['mnap', 'hit_rate', 'mrr']].mean()
        exploded_predict: pd.Series = predict.explode()
        metric_vals['coverage'] = calc_coverage(exploded_predict, self._data.organisations)
        metric_vals['surprisal'] = calc_surprisal(exploded_predict, x_train)
        return metric_vals

    def score(self, model, experiment_name=None):
        if experiment_name is None:
            experiment_name = secrets.token_urlsafe(8)
        X = self._data.reviews
        metrics = defaultdict(list)
        for train_index, test_index in self._cv_splitter.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            fitted_model = copy(model).fit(X_train)
            predict = fitted_model.predict(X_test[['user_id']].drop_duplicates())
            current_fold_score = self.scoring_fn(predict, X_train, X_test)
            for key, value in current_fold_score.items():
                metrics[key].append(value)
        log_dict = {
            'name': experiment_name,
            'model': str(model),
        }
        for key, value in metrics.items():
            mean_score = np.mean(value)
            std_score = np.std(value)
            log_dict.update(
                {
                    key + '_mean': mean_score,
                    key + '_std': std_score
                }
            )
        self._score_table = self._score_table._append(log_dict, ignore_index=True)
        return log_dict