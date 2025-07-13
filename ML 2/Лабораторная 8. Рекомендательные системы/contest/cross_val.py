import numpy as np
from sklearn.model_selection._split import _BaseKFold

class DaysTimeSeriesSplit(_BaseKFold):
    def __init__(self, n_splits, test_size_in_days, min_train_size_in_days, days_column_name='ts'):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.test_size_in_days = test_size_in_days
        self.min_train_size_in_days = min_train_size_in_days
        self.days_column_name = days_column_name

    def split(self, X, y=None, groups=None):
        X = X.reset_index()
        unique_days = np.sort(X[self.days_column_name].unique())
        n_days = len(unique_days)
        min_train_days = self.min_train_size_in_days
        test_days = self.test_size_in_days
        max_splits = self.n_splits

        if n_days < min_train_days + test_days:
            raise ValueError("Not enough days in the dataset")

        # Считаем, сколько дней можно сдвигать окно данных
        possible_starts = n_days - min_train_days - test_days + 1
        if max_splits > possible_starts + 1:  # +1 — последний фолд покрывает весь датасет
            raise ValueError("Too many splits for such a short time range")

        fold_starts = np.linspace(0, possible_starts,
                                  num=max_splits, endpoint=False, dtype=int)
        for i, start in enumerate(fold_starts):
            if i == max_splits - 1:
                # Последний фолд: train - всё до теста, тест - последние test_days
                train_days = unique_days[:-test_days]
                test_days_arr = unique_days[-test_days:]
            else:
                train_days = unique_days[:min_train_days + start]
                test_days_arr = unique_days[min_train_days + start: min_train_days + start + test_days]
            train_idx = X[X[self.days_column_name].isin(train_days)].index.values
            test_idx = X[X[self.days_column_name].isin(test_days_arr)].index.values
            yield train_idx, test_idx




