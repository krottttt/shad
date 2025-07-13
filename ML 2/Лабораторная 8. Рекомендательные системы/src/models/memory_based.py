import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix

from src.models.based import EstimatorWithFallback
from src.models.statistics_based import PopularityRecommender


def cosine_similarity(matrix_1, matrix_2):
    """
    cosine similarity between two sparse matrices <u, v> / ||u|| / ||v||
    :param matrix_1: sparse matrix of shape (n_users, n_items)
    :param matrix_2: sparse matrix of shape (n_users, n_items)
    :return: sparse matrix of similarities
    """
    # Вычисляем произведение <u, v>
    numerator = matrix_1.dot(matrix_2.T).toarray()

    # Вычисляем ||u|| для всех строк matrix_1
    norm_1 = np.sqrt(np.array(matrix_1.power(2).sum(axis=1)).flatten())

    # Вычисляем ||v|| для всех строк matrix_2
    norm_2 = np.sqrt(np.array(matrix_2.power(2).sum(axis=1)).flatten())

    # Создаем матрицы нормировки
    norm_1_matrix = norm_1.reshape(-1, 1)
    norm_2_matrix = norm_2.reshape(1, -1)

    # Избегаем деления на ноль
    denominator = norm_1_matrix.dot(norm_2_matrix)
    denominator[denominator == 0] = 1  # Чтобы избежать деления на ноль

    # Вычисляем косинусное сходство
    similarity = numerator / denominator

    # Нормализуем сходство, чтобы сумма абсолютных значений равнялась 1 для каждой строки
    row_sums = np.sum(np.abs(similarity), axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Чтобы избежать деления на ноль
    normalized_similarity = similarity / row_sums

    # Обнуляем диагональ (self-similarity)
    np.fill_diagonal(normalized_similarity, 0)

    return csr_matrix(normalized_similarity)


def pearson_similarity(matrix_1, matrix_2):
    """
    pearson similarity between two sparse matrices <u - Eu, v - Ev> / Vu / Vv
    :param matrix_1: sparse matrix of shape (n_users, n_items)
    :param matrix_2: sparse matrix of shape (n_users, n_items)
    :return: sparse matrix of similarities
    """
    # Получаем средние рейтинги для каждого пользователя
    mask_1 = matrix_1.copy()
    mask_1.data = np.ones_like(mask_1.data)
    counts_1 = np.array(mask_1.sum(axis=1)).flatten()
    counts_1[counts_1 == 0] = 1  # Избегаем деления на ноль
    means_1 = np.array(matrix_1.sum(axis=1)).flatten() / counts_1

    mask_2 = matrix_2.copy()
    mask_2.data = np.ones_like(mask_2.data)
    counts_2 = np.array(mask_2.sum(axis=1)).flatten()
    counts_2[counts_2 == 0] = 1  # Избегаем деления на ноль
    means_2 = np.array(matrix_2.sum(axis=1)).flatten() / counts_2

    # Центрируем данные
    centered_1 = matrix_1.copy()
    for i in range(matrix_1.shape[0]):
        if centered_1[i].nnz > 0:  # Если строка не пустая
            centered_1[i] = centered_1[i] - means_1[i]

    centered_2 = matrix_2.copy()
    for i in range(matrix_2.shape[0]):
        if centered_2[i].nnz > 0:  # Если строка не пустая
            centered_2[i] = centered_2[i] - means_2[i]

    # Вычисляем числитель: <u - Eu, v - Ev>
    numerator = centered_1.dot(centered_2.T).toarray()

    # Вычисляем знаменатель: Vu * Vv
    var_1 = np.sqrt(np.array(centered_1.power(2).sum(axis=1)).flatten())
    var_2 = np.sqrt(np.array(centered_2.power(2).sum(axis=1)).flatten())

    var_1_matrix = var_1.reshape(-1, 1)
    var_2_matrix = var_2.reshape(1, -1)

    denominator = var_1_matrix.dot(var_2_matrix)
    denominator[denominator == 0] = 1  # Избегаем деления на ноль

    # Вычисляем корреляцию Пирсона
    similarity = numerator / denominator

    # Нормализуем сходство, чтобы сумма абсолютных значений равнялась 1 для каждой строки
    row_sums = np.sum(np.abs(similarity), axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Избегаем деления на ноль
    normalized_similarity = similarity / row_sums

    # Обнуляем диагональ (self-similarity)
    np.fill_diagonal(normalized_similarity, 0)

    return csr_matrix(normalized_similarity)


class UserBasedRecommender(EstimatorWithFallback):
    def __init__(self, similarity_measure, n_items, fallback_estimator=PopularityRecommender, **kwargs):
        super().__init__(fallback_estimator, n_items=n_items, **kwargs)
        self.similarity_measure = similarity_measure
        if self.similarity_measure == 'cosine':
            self.similarity_fn = cosine_similarity
        elif self.similarity_measure == 'pearson':
            self.similarity_fn = pearson_similarity
        else:
            raise NotImplementedError(f'similarity measure {self.similarity_measure} is not implemented')
        self.n_items = n_items

    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X: DataFrame with reviews
        :return: sparse matrix of ratings
        """
        # Создаем маппинг для user_id и org_id
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(X['user_id'].unique())}
        self.org_id_to_idx = {org_id: idx for idx, org_id in enumerate(X['org_id'].unique())}

        # Обратное преобразование
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.idx_to_org_id = {idx: org_id for org_id, idx in self.org_id_to_idx.items()}

        # Создаем разреженную матрицу рейтингов
        rows = [self.user_id_to_idx[user_id] for user_id in X['user_id']]
        cols = [self.org_id_to_idx[org_id] for org_id in X['org_id']]
        data = X['rating'].values

        n_users = len(self.user_id_to_idx)
        n_orgs = len(self.org_id_to_idx)

        return csr_matrix((data, (rows, cols)), shape=(n_users, n_orgs))

    def fit(self, X, y=None):
        super().fit(X)
        self._x = self.make_sparse(X)
        return self

    def select_users(self, matrix, users):
        """helper function which selects users, e.g. zeroes out the unused part of matrix"""
        # Преобразуем user_id в индексы
        user_indices = [self.user_id_to_idx.get(user_id, -1) for user_id in users]

        # Отфильтровываем неизвестные user_id
        valid_indices = [idx for idx in user_indices if idx != -1]

        # Создаем новую матрицу только с выбранными пользователями
        if valid_indices:
            return matrix[valid_indices, :]
        else:
            # Возвращаем пустую матрицу с правильной размерностью по столбцам
            return csr_matrix((0, matrix.shape[1]))

    def select_orgs(self, matrix, orgs):
        """helper function which selects orgs, e.g. zeroes out the unused part of matrix"""
        # Преобразуем org_id в индексы
        org_indices = [self.org_id_to_idx.get(org_id, -1) for org_id in orgs]

        # Отфильтровываем неизвестные org_id
        valid_indices = [idx for idx in org_indices if idx != -1]

        # Создаем новую матрицу только с выбранными организациями
        if valid_indices:
            return matrix[:, valid_indices]
        else:
            # Возвращаем пустую матрицу с правильной размерностью по строкам
            return csr_matrix((matrix.shape[0], 0))

    def predict_user_org(self, users, orgs):
        rating = self.compute_rating(self._x, orgs, users)
        rating.eliminate_zeros()
        if rating.nnz > 0:
            ranking_df = rating.tocoo()
            ranking_df = pd.DataFrame({'rating': ranking_df.data, 'user_id': ranking_df.row, 'org_id': ranking_df.col})
            prediction = ranking_df.groupby('user_id').apply(
                lambda group: group.nlargest(self.n_items, columns='rating')['org_id'].values, include_groups=False)
        else:
            prediction = pd.Series()
        return prediction

    def compute_rating(self, matrix, orgs, users):
        """
        compute the actual rating given by similar users weighted by their similarity (only for interactions)
        :param matrix: sparse matrix of ratings
        :param orgs: list of org_ids to recommend
        :param users: list of user_ids to recommend to
        :return: sparse matrix of predicted ratings
        """
        # Преобразуем идентификаторы пользователей и организаций в индексы
        user_indices = [self.user_id_to_idx.get(user_id, -1) for user_id in users]
        valid_user_indices = [idx for idx in user_indices if idx != -1]

        org_indices = [self.org_id_to_idx.get(org_id, -1) for org_id in orgs]
        valid_org_indices = [idx for idx in org_indices if idx != -1]

        if not valid_user_indices or not valid_org_indices:
            return csr_matrix((len(users), len(orgs)))

        # Выбираем подматрицу для наших пользователей
        user_matrix = matrix[valid_user_indices, :]

        # Вычисляем сходство между пользователями
        similarity_matrix = self.similarity_fn(user_matrix, matrix)

        # Вычисляем предсказанные рейтинги как взвешенную сумму рейтингов похожих пользователей
        predicted_ratings = similarity_matrix.dot(matrix[:, valid_org_indices])

        # Создаем матрицу результатов с отображением на исходные индексы
        result = csr_matrix((len(users), len(orgs)))

        # Заполняем матрицу результатов
        for i, user_idx in enumerate(valid_user_indices):
            for j, org_idx in enumerate(valid_org_indices):
                user_pos = user_indices.index(user_idx)
                org_pos = org_indices.index(org_idx)
                result[user_pos, org_pos] = predicted_ratings[i, j]

        return result


class ItemBasedRecommender(UserBasedRecommender):
    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X: DataFrame with reviews
        :return: sparse matrix of ratings (items in rows, users in columns)
        """
        # Создаем маппинг для user_id и org_id
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(X['user_id'].unique())}
        self.org_id_to_idx = {org_id: idx for idx, org_id in enumerate(X['org_id'].unique())}

        # Обратное преобразование
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.idx_to_org_id = {idx: org_id for org_id, idx in self.org_id_to_idx.items()}

        # Создаем разреженную матрицу рейтингов (org_id x user_id - транспонированную)
        rows = [self.org_id_to_idx[org_id] for org_id in X['org_id']]
        cols = [self.user_id_to_idx[user_id] for user_id in X['user_id']]
        data = X['rating'].values

        n_orgs = len(self.org_id_to_idx)
        n_users = len(self.user_id_to_idx)

        return csr_matrix((data, (rows, cols)), shape=(n_orgs, n_users))

    def select_users(self, matrix, users):
        """
        helper function which selects users from transposed matrix
        :param matrix: sparse matrix (orgs x users)
        :param users: list of user_ids
        :return: sparse submatrix
        """
        # Преобразуем user_id в индексы
        user_indices = [self.user_id_to_idx.get(user_id, -1) for user_id in users]

        # Отфильтровываем неизвестные user_id
        valid_indices = [idx for idx in user_indices if idx != -1]

        # Создаем новую матрицу только с выбранными пользователями (в столбцах)
        if valid_indices:
            return matrix[:, valid_indices]
        else:
            # Возвращаем пустую матрицу с правильной размерностью по строкам
            return csr_matrix((matrix.shape[0], 0))

    def select_orgs(self, matrix, orgs):
        """
        helper function which selects orgs from transposed matrix
        :param matrix: sparse matrix (orgs x users)
        :param orgs: list of org_ids
        :return: sparse submatrix
        """
        # Преобразуем org_id в индексы
        org_indices = [self.org_id_to_idx.get(org_id, -1) for org_id in orgs]

        # Отфильтровываем неизвестные org_id
        valid_indices = [idx for idx in org_indices if idx != -1]

        # Создаем новую матрицу только с выбранными организациями (в строках)
        if valid_indices:
            return matrix[valid_indices, :]
        else:
            # Возвращаем пустую матрицу с правильной размерностью по столбцам
            return csr_matrix((0, matrix.shape[1]))

    def compute_rating(self, matrix, orgs, users):
        """
        compute the actual rating given by similar items weighted by their similarity
        :param matrix: sparse matrix of ratings (orgs x users)
        :param orgs: list of org_ids to recommend
        :param users: list of user_ids to recommend to
        :return: sparse matrix of predicted ratings (users x orgs)
        """
        # Преобразуем идентификаторы пользователей и организаций в индексы
        user_indices = [self.user_id_to_idx.get(user_id, -1) for user_id in users]
        valid_user_indices = [idx for idx in user_indices if idx != -1]

        org_indices = [self.org_id_to_idx.get(org_id, -1) for org_id in orgs]
        valid_org_indices = [idx for idx in org_indices if idx != -1]

        if not valid_user_indices or not valid_org_indices:
            return csr_matrix((len(users), len(orgs)))

        # Получаем матрицу взаимодействий пользователей (для вычисления, какие организации уже оценены)
        user_item_matrix = matrix[:, valid_user_indices].T

        # Выбираем подматрицу для целевых организаций
        target_orgs_matrix = matrix[valid_org_indices, :]

        # Вычисляем сходство между организациями
        similarity_matrix = self.similarity_fn(target_orgs_matrix, matrix)

        # Вычисляем предсказанные рейтинги как взвешенную сумму рейтингов похожих организаций
        # Результат будет иметь размерность (len(valid_org_indices), len(valid_user_indices))
        predicted_ratings = similarity_matrix.dot(matrix[:, valid_user_indices]).T

        # Создаем матрицу результатов с отображением на исходные индексы
        result = csr_matrix((len(users), len(orgs)))

        # Заполняем матрицу результатов
        for i, user_idx in enumerate(valid_user_indices):
            for j, org_idx in enumerate(valid_org_indices):
                user_pos = user_indices.index(user_idx)
                org_pos = org_indices.index(org_idx)
                result[user_pos, org_pos] = predicted_ratings[i, j]

        return result

