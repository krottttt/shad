import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, linalg as splinalg
from sklearn.utils.extmath import randomized_svd

from src.models.based import EstimatorWithFallback
from src.models.statistics_based import PopularityRecommender


class SVDRecommender(EstimatorWithFallback):
    def __init__(self, n_components, n_items, fallback_estimator=PopularityRecommender, random_state=None):
        super().__init__(fallback_estimator, n_items=n_items)
        self.n_components = n_components
        self._random_state = random_state
        self.n_items = n_items
        self.rng = np.random.default_rng(random_state)

    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X: DataFrame with reviews
        :return: sparse matrix of ratings, user and org mappings
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
        """
        compute user and item representations using (presumably) sparse svd
        :param X: DataFrame with reviews
        :param y: Not used
        :return: self
        """
        super().fit(X)

        # Создаем разреженную матрицу рейтингов
        self.rating_matrix = self.make_sparse(X)

        # Вычисляем SVD разложение
        U, sigma, VT = randomized_svd(
            self.rating_matrix,
            n_components=min(self.n_components, min(self.rating_matrix.shape) - 1),
            random_state=self._random_state
        )

        # Сохраняем сингулярные значения и векторы
        self.U = U
        self.sigma = sigma
        self.VT = VT

        # Для прогнозирования рейтингов мы будем использовать U * sigma и VT
        self.user_factors = U * sigma
        self.item_factors = VT.T

        return self

    def predict_user_org(self, users, orgs):
        """
        use embeddings to compute user predictions (ordered by reconstructed score)
        :param users: predict for these users
        :param orgs: use orgs from this set
        :return: Series with predictions for each user
        """
        # Преобразуем пользователей и организации в индексы
        user_indices = [self.user_id_to_idx.get(user_id) for user_id in users]
        org_indices = [self.org_id_to_idx.get(org_id) for org_id in orgs]

        # Отфильтровываем отсутствующие индексы
        valid_user_indices = [(i, idx) for i, idx in enumerate(user_indices) if idx is not None]
        valid_org_indices = [(i, idx) for i, idx in enumerate(org_indices) if idx is not None]

        if not valid_user_indices or not valid_org_indices:
            # Если нет данных для прогнозирования, возвращаем пустую серию
            return pd.Series(index=users, dtype=object)

        # Создаем массив для хранения предсказаний
        predictions = []

        # Для каждого пользователя вычисляем предсказанные рейтинги для организаций
        for user_idx, matrix_user_idx in valid_user_indices:
            # Получаем вектор факторов пользователя
            user_vector = self.user_factors[matrix_user_idx]

            # Вычисляем рейтинги для всех организаций
            org_ratings = []
            for org_idx, matrix_org_idx in valid_org_indices:
                # Получаем вектор факторов организации
                org_vector = self.item_factors[matrix_org_idx]

                # Вычисляем рейтинг как скалярное произведение
                rating = np.dot(user_vector, org_vector)
                org_ratings.append((orgs[org_idx], rating))

            # Сортируем организации по рейтингу и берем top-n
            top_orgs = sorted(org_ratings, key=lambda x: x[1], reverse=True)[:self.n_items]
            predictions.append((users[user_idx], [org for org, _ in top_orgs]))

        # Создаем Series с предсказаниями
        return pd.Series(dict(predictions), name='prediction')


class ALSRecommender(EstimatorWithFallback):
    def __init__(self, n_items, feature_dim, regularizer, num_iter, fallback_estimator=PopularityRecommender,
                 random_state=None):
        super().__init__(fallback_estimator, n_items=n_items)
        self.n_items = n_items
        self.feature_dim = feature_dim
        self.regularizer = regularizer
        self.num_iter = num_iter
        self._random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def update_other_embeddings(self, embeddings, rating_matrix, lambda_reg):
        """
        compute alternate set of embeddings
        :param embeddings: matrix [n_entries, n_features]
        :param rating_matrix: sparse representation of interactions
        :param lambda_reg: regularization parameter
        :return: updated embeddings
        """
        n_items = rating_matrix.shape[1]
        n_features = embeddings.shape[1]
        result = np.zeros((n_items, n_features))

        # Для каждого item обновляем его embedding
        for i in range(n_items):
            # Получаем оценки для этого item
            users_idx = rating_matrix[:, i].nonzero()[0]

            if len(users_idx) > 0:
                # Получаем эмбеддинги пользователей, которые оценили item
                user_embeddings = embeddings[users_idx]
                # Получаем оценки, которые эти пользователи поставили
                ratings = rating_matrix[users_idx, i].toarray().flatten()

                # Вычисляем новый эмбеддинг для item с помощью метода наименьших квадратов с регуляризацией
                A = user_embeddings.T @ user_embeddings + lambda_reg * np.eye(n_features)
                b = user_embeddings.T @ ratings

                result[i] = np.linalg.solve(A, b)

        return result

    def compute_loss(self, user_embeddings, item_embeddings, rating_matrix):
        """
        compute reconstruction and regularizing loss combination
        :param user_embeddings: matrix [n_users, n_features]
        :param item_embeddings: matrix [n_items, n_features]
        :param rating_matrix: sparse representation of interactions
        :return: total loss value
        """
        # Вычисляем реконструкционную ошибку
        reconstruction_loss = 0
        n_ratings = 0

        # Перебираем все ненулевые элементы в разреженной матрице
        rows, cols = rating_matrix.nonzero()
        for i, j in zip(rows, cols):
            # Получаем истинный рейтинг
            true_rating = rating_matrix[i, j]

            # Получаем эмбеддинги пользователя и item
            user_emb = user_embeddings[i]
            item_emb = item_embeddings[j]

            # Вычисляем предсказанный рейтинг
            pred_rating = np.dot(user_emb, item_emb)

            # Накапливаем ошибку
            reconstruction_loss += (true_rating - pred_rating) ** 2
            n_ratings += 1

        # Если нет рейтингов, возвращаем бесконечность
        if n_ratings == 0:
            return float('inf')

        # Вычисляем регуляризационную ошибку
        user_reg_loss = self.regularizer * np.sum(np.square(user_embeddings))
        item_reg_loss = self.regularizer * np.sum(np.square(item_embeddings))

        # Общая ошибка
        total_loss = reconstruction_loss + user_reg_loss + item_reg_loss

        return total_loss / n_ratings

    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X: DataFrame with reviews
        :return: sparse matrix of ratings, user and org mappings
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
        """
        tune embeddings for self.n_iter iterations, record loss history
        :param X: DataFrame with reviews
        :param y: Not used
        :return: self
        """
        super().fit(X)
        self._history = []

        # Создаем разреженную матрицу рейтингов
        self.rating_matrix = self.make_sparse(X)
        n_users, n_items = self.rating_matrix.shape

        # Инициализируем эмбеддинги случайными значениями
        self.user_embeddings = self.rng.standard_normal((n_users, self.feature_dim))
        self.item_embeddings = self.rng.standard_normal((n_items, self.feature_dim))

        # Нормализуем начальные эмбеддинги
        self.user_embeddings /= np.sqrt(self.feature_dim)
        self.item_embeddings /= np.sqrt(self.feature_dim)

        # Выполняем итерации алгоритма ALS
        for iter_num in range(self.num_iter):
            # Обновляем эмбеддинги организаций, фиксируя эмбеддинги пользователей
            self.item_embeddings = self.update_other_embeddings(
                self.user_embeddings, self.rating_matrix, self.regularizer)

            # Обновляем эмбеддинги пользователей, фиксируя эмбеддинги организаций
            self.user_embeddings = self.update_other_embeddings(
                self.item_embeddings, self.rating_matrix.T, self.regularizer)

            # Вычисляем и сохраняем текущую ошибку
            loss = self.compute_loss(self.user_embeddings, self.item_embeddings, self.rating_matrix)
            self._history.append(loss)

        return self

    def predict_user_org(self, users, orgs):
        """
        use embeddings to compute user predictions (ordered by reconstructed score)
        :param users: predict for these users
        :param orgs: use orgs from this set
        :return: Series with predictions for each user
        """
        # Преобразуем пользователей и организации в индексы
        user_indices = [self.user_id_to_idx.get(user_id) for user_id in users]
        org_indices = [self.org_id_to_idx.get(org_id) for org_id in orgs]

        # Отфильтровываем отсутствующие индексы
        valid_user_indices = [(i, idx) for i, idx in enumerate(user_indices) if idx is not None]
        valid_org_indices = [(i, idx) for i, idx in enumerate(org_indices) if idx is not None]

        if not valid_user_indices or not valid_org_indices:
            # Если нет данных для прогнозирования, возвращаем пустую серию
            return pd.Series(index=users, dtype=object)

        # Создаем массив для хранения предсказаний
        predictions = []

        # Для каждого пользователя вычисляем предсказанные рейтинги для организаций
        for user_idx, matrix_user_idx in valid_user_indices:
            # Получаем вектор факторов пользователя
            user_vector = self.user_embeddings[matrix_user_idx]

            # Вычисляем рейтинги для всех организаций
            org_ratings = []
            for org_idx, matrix_org_idx in valid_org_indices:
                # Получаем вектор факторов организации
                org_vector = self.item_embeddings[matrix_org_idx]

                # Вычисляем рейтинг как скалярное произведение
                rating = np.dot(user_vector, org_vector)
                org_ratings.append((orgs[org_idx], rating))

            # Сортируем организации по рейтингу и берем top-n
            top_orgs = sorted(org_ratings, key=lambda x: x[1], reverse=True)[:self.n_items]
            predictions.append((users[user_idx], [org for org, _ in top_orgs]))

        # Создаем Series с предсказаниями
        return pd.Series(dict(predictions), name='prediction')

