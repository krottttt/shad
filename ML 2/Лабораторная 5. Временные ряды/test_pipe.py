
import datetime
import sklearn
import typing as tp
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X_type = tp.NewType("X_type", np.ndarray)
X_row_type = tp.NewType("X_row_type", np.ndarray)
Y_type = tp.NewType("Y_type", np.array)
TS_type = tp.NewType("TS_type", pd.Series)
Model_type = tp.TypeVar("Model_type")


def read_timeseries(path_to_df: str = "train.csv") -> TS_type:
    """Функция для чтения данных и получения обучающей и тестовой выборок"""
    df = pd.read_csv(path_to_df)
    df = df[(df['store'] == 1) & (df['item'] == 1)]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    ts = df["sales"]
    train_ts = ts[:-365]
    test_ts = ts[-365:]
    return train_ts, test_ts


def extract_hybrid_strategy_features(
        timeseries: TS_type,
        model_idx: int,
        window_size: int = 7
) -> X_row_type:
    """
    Функция для получения вектора фичей согласно гибридной схеме. На вход подаётся временной ряд
    до момента T, функция выделяет из него фичи, необходимые модели под номером model_idx для
    прогноза на момент времени T

    Args:
        timeseries --- временной ряд до момента времени T (не включительно), pd.Series с датой
                       в качестве индекса
        model_idx --- индекс модели, то есть номер шага прогноза,
                      для которого нужно получить признаки, нумерация с нуля
        window_size --- количество последних значений ряда, используемых для прогноза
                        (без учёта количества прогнозов с предыдущих этапов)

    Returns:
        Одномерный вектор фичей для модели с индексом model_idx (np.array),
        чтобы сделать прогноз для момента времени T
    """
    feature_window = window_size + model_idx
    if feature_window == 0:
        return np.array([])
    else:
        return timeseries[-feature_window:].values


def is_retail_important_day(date):
    # Словарь с важными для розничной торговли датами
    retail_dates = {
        # Основные праздники
        "new_year": (date.month == 1 and date.day == 1),
        "christmas": (date.month == 12 and date.day == 25),
        "valentines": (date.month == 2 and date.day == 14),
        "halloween": (date.month == 10 and date.day == 31),

        # Сезонные особенности
        "back_to_school": (date.month == 8 and date.day >= 15) or (date.month == 9 and date.day <= 15),
        "summer_start": (date.month == 6 and date.day <= 10),
        "winter_sale": (date.month == 1 and date.day >= 5 and date.day <= 15),
    }

    return [1 if retail_dates[key] else 0 for key in retail_dates]

def extract_advanced_features(
        timeseries: TS_type,
        model_idx: int,
        window_size: int = 21,
        advanced: bool = False
) -> X_row_type:
    """
    Расширенная функция для получения вектора фичей с дополнительными признаками

    Args:
        timeseries --- временной ряд до момента времени T (не включительно)
        model_idx --- индекс модели
        window_size --- количество последних значений ряда для прогноза

    Returns:
        Расширенный вектор фичей для модели с индексом model_idx
    """
    # Базовые признаки
    base_features = extract_hybrid_strategy_features(timeseries, model_idx, window_size)

    # Если недостаточно данных, возвращаем только базовые признаки
    if len(timeseries) < window_size + model_idx or not(advanced):
        return base_features

    feature_window = window_size + model_idx
    # Добавляем признаки на основе дат
    dates = timeseries.index[-feature_window:] if isinstance(timeseries, pd.Series) else pd.to_datetime(datetime.datetime.now())
    date_features = []

    for date in dates:
        date_features.extend([
            date.dayofweek,  # День недели
            date.month,  # Месяц
            date.day,  # День месяца
            date.quarter,  # Квартал
            # Признак выходного дня
            1 if date.dayofweek >= 5 else 0
        ])
    next_date = dates[-1]+ pd.Timedelta(days=1)
    date_features.extend([
            next_date.dayofweek,  # День недели
            next_date.month,  # Месяц
            next_date.day,  # День месяца
            next_date.quarter,  # Квартал
            # Признак выходного дня
            1 if next_date.dayofweek >= 5 else 0
        ])
    date_features.extend(is_retail_important_day(next_date))

    trend_features = []

    # Добавляем статистические признаки
    if len(timeseries) >= window_size + model_idx:
        recent_data = timeseries[-window_size - model_idx:].values
        stat_features = [
            np.mean(recent_data),  # Среднее
            np.std(recent_data),  # Стандартное отклонение
            np.min(recent_data),  # Минимум
            np.max(recent_data),  # Максимум
            np.median(recent_data)  # Медиана
        ]
        diff = np.diff(recent_data)
        trend_features.extend([
            np.mean(diff),
            np.std(diff) if len(diff) > 1 else 0,
            np.median(diff) if len(diff) > 1 else diff[0],
            np.min(diff),
            np.max(diff),
            np.sum(diff > 0) / len(diff),  # Доля положительных изменений
        ])
        for window in [3, 7, 14]:
            if len(recent_data) >= window:
                stat_features.append(np.mean(recent_data[-window:]))

        # # Добавляем лаги для учета сезонности
        # if len(timeseries) >= window_size + model_idx + 7:
        #     weekly_lag = timeseries[-(window_size + model_idx + 7):-(model_idx + 7)].values
        #     stat_features.extend([np.mean(weekly_lag), np.std(weekly_lag)])
        # else:
        #     stat_features.extend([0, 0])
    else:
        stat_features = [0, 0, 0, 0, 0, 0, 0]

    # Объединяем все признаки
    all_features = np.concatenate([base_features, date_features, stat_features])
    return all_features



# def build_datasets(
#         timeseries: TS_type,
#         extract_features: tp.Callable[..., X_row_type],
#         window_size: int,
#         model_count: int
# ) -> tp.List[tp.Tuple[X_type, Y_type]]:
#     """
#     Функция для получения обучающих датасетов согласно гибридной схеме
#
#     Args:
#         timeseries --- временной ряд
#         extract_features --- функция для генерации вектора фичей
#         window_size --- количество последних значений ряда, используемых для прогноза
#         model_count --- количество моделей, используемых для получения предскзаний
#
#     Returns:
#         Список из model_count датасетов, i-й датасет используется для обучения i-й модели
#         и представляет собой пару из двумерного массива фичей и одномерного массива таргетов
#     """
#     datasets = []
#     n = len(timeseries)
#     X = [[] for i in range(model_count)]
#     y = [[] for i in range(model_count)]
#     for j in range(n - window_size):
#         ts = extract_features(timeseries[j:j+model_count+window_size], model_count, window_size)
#         if n - window_size - j >= 3:
#             for i in range(model_count):
#                 X[i].append(ts[:i+window_size])
#                 y[i].append(ts[i+window_size])
#         else:
#             for i in range(n - window_size - j):
#                 X[i].append(ts[:i+window_size])
#
#                 y[i].append(ts[i+window_size])
#
#     for i in range(model_count):
#         datasets.append((np.array(X[i]), np.array(y[i])))
#
#     assert len(datasets) == model_count
#     return datasets

def build_datasets(
        timeseries: TS_type,
        extract_features: tp.Callable[..., X_row_type],
        window_size: int,
        model_count: int,
        advanced: bool = False
) -> tp.List[tp.Tuple[X_type, Y_type]]:
    """
    Функция для получения обучающих датасетов согласно гибридной схеме

    Args:
        timeseries --- временной ряд
        extract_features --- функция для генерации вектора фичей
        window_size --- количество последних значений ряда, используемых для прогноза
        model_count --- количество моделей, используемых для получения предскзаний

    Returns:
        Список из model_count датасетов, i-й датасет используется для обучения i-й модели
        и представляет собой пару из двумерного массива фичей и одномерного массива таргетов
    """
    datasets = []

    for i in range(model_count):
        X, y = [], []

        # Формируем обучающую выборку для i-й модели
        for j in range(window_size + i, len(timeseries)):
            # Получаем фичи для текущей точки, учитывая индекс модели
            features = extract_features(timeseries[:j], i, window_size,advanced=advanced)
            X.append(features)
            y.append(timeseries[j])

        datasets.append((np.array(X), np.array(y)))

    assert len(datasets) == model_count
    return datasets



def predict(
        timeseries: TS_type,
        models: tp.List[Model_type],
        extract_features: tp.Callable[..., X_row_type] = extract_advanced_features,
        advanced: bool = False
) -> TS_type:
    """
    Функция для получения прогноза len(models) следующих значений временного ряда

    Args:
        timeseries --- временной ряд, по которому необходимо сделать прогноз на следующие даты
        models --- список обученных моделей, i-я модель используется для получения i-го прогноза
        extract_features --- функция для генерации вектора фичей. Если вы реализуете свою функцию
                             извлечения фичей для конечной модели, передавайте этим аргументом.
                             Внутри функции predict функцию extract_features нужно вызывать только
                             с аргументами timeseries и model_idx, остальные должны быть со значениями
                             по умолчанию

    Returns:
        Прогноз len(models) следующих значений временного ряда
    """
    predictions = []
    current_ts = timeseries.copy()

    for i, model in enumerate(models):
        # Получаем признаки для текущего шага прогноза
        features = extract_features(current_ts, i,advanced=advanced)

        # Преобразуем в 2D массив для sklearn моделей
        features_2d = features.reshape(1, -1)

        # Делаем прогноз
        prediction = model.predict(features_2d)[0]
        predictions.append(prediction)

        # Добавляем прогноз в временной ряд для следующего шага
        next_date = current_ts.index[-1] + pd.Timedelta(days=1)
        prediction_series = pd.Series([prediction], index=[next_date])
        current_ts = pd.concat([current_ts, prediction_series])

    return np.array(predictions)




def train_models(
        train_timeseries: TS_type,
        model_count: int
) -> tp.List[Model_type]:
    """
    Функция для получения обученных моделей

    Args:
        train_timeseries --- обучающий временной ряд
        model_count --- количество моделей для обучения согласно гибридной схеме.
                        Прогнозирование должно выполняться на model_count дней вперёд

    Returns:
        Список из len(datasets) обученных моделей
    """
    window_size = 21  # Увеличенный размер окна для улучшения качества прогноза

    # Используем расширенные признаки для лучшего прогноза
    datasets = build_datasets(train_timeseries, extract_advanced_features, window_size, model_count,advanced = True)

    models = []

    for i, (X, y) in enumerate(datasets):
        # Выбираем разные модели в зависимости от горизонта прогноза
        if i < 7:  # Ближайшая неделя
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42 + i
                ))
            ])
        elif i < 14:  # Следующая неделя
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42 + i
                ))
            ])
        else:  # Долгосрочный прогноз
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(
                    alpha=1.0,
                    random_state=42 + i
                ))
            ])

        model.fit(X, y)
        models.append(model)

    assert len(models) == len(datasets)
    return models
    #
    # models = []
    #
    # # datasets = build_datasets(train_timeseries, ...)
    # # YOUR CODE HERE
    #
    # assert len(models) == len(datasets)
    # return models



def score_models(
        train_ts: TS_type,
        test_ts: TS_type,
        models: tp.List[Model_type],
        predict: tp.Callable[[TS_type, tp.List[Model_type]], TS_type] = predict,
        extract_features: tp.Callable[..., X_row_type] = extract_advanced_features
):
    """
    Функция для оценки качества обученных моделей по метрике MSE

    Args:
        train_ts --- обучающий временной ряд
        test_ts --- тестовый временной ряд
        models --- список обученных моделей
        predict --- функция для получения прогноза временного ряда

    Returns:
        Усредненное MSE для прогноза моделей по всей тестовой выборке
    """
    predict_len = len(models)
    predictions = []
    targets = []

    for i in range(len(test_ts) - predict_len + 1):
        predictions.extend(list(predict(train_ts, models,extract_features,True)))
        targets.extend(list(test_ts[i:i + predict_len]))
        train_ts = pd.concat([train_ts, test_ts[i:i + 1]])

    return sklearn.metrics.mean_squared_error(targets, predictions)
