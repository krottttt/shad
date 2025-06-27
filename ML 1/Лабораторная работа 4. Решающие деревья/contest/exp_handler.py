import json
import contextlib
import os
import pathlib
import typing as tp
import uuid

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.model_selection import cross_validate,cross_val_predict







class Logger:
    """Logger performs data management and stores scores and other relevant information"""

    def __init__(self, logs_path: tp.Union[str, os.PathLike]):
        self.path = pathlib.Path(logs_path)

        records = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith('.json'):
                    uuid = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            logged_data = json.load(f)
                            records.append(
                                {
                                    'id': uuid,
                                    **logged_data
                                }
                            )
                        except json.JSONDecodeError:
                            pass
        if records:
            self.leaderboard = pd.DataFrame.from_records(records, index='id')
        else:
            self.leaderboard = pd.DataFrame(index=pd.Index([], name='id'))

        self._current_run = None

    class Run:
        """Run incapsulates information for a particular entry of logged material. Each run is solitary experiment"""

        def __init__(self, name, storage, path):
            self.name = name
            self._storage = storage
            self._path = path
            self._storage.append(pd.Series(name=name))

        def log(self, key, value):
            self._storage.loc[self.name, key] = value

        def log_values(self, log_values: tp.Dict[str, tp.Any]):
            for key, value in log_values.items():
                self.log(key, value)

        def save_logs(self):
            with open(self._path / f'{self.name}.json', 'w+') as f:
                json.dump(self._storage.loc[self.name].to_dict(), f)

        def log_artifact(self, fname: str, writer: tp.Callable):
            with open(self._path / fname, 'wb+') as f:
                writer(f)

    @contextlib.contextmanager
    def run(self, name: tp.Optional[str] = None):
        if name is None:
            name = str(uuid.uuid4())
        elif name in self.leaderboard.index:
            raise NameError("Run with given name already exists, name should be unique")
        else:
            name = name.replace(' ', '_')
        self._current_run = Logger.Run(name, self.leaderboard, self.path / name)
        os.makedirs(self.path / name, exist_ok=True)
        try:
            yield self._current_run
        finally:
            self._current_run.save_logs()


def load_predictions_dataframe(filename, column_prefix, index):
    with open(filename, 'rb') as file:
        data = np.load(file)
        dataframe = pd.DataFrame(data, columns=[f'{column_prefix}_{i}' for i in range(data.shape[1])],
                                 index=index)
        return dataframe


class ExperimentHandler:
    """This class perfoms experiments with given model, measures metrics and logs everything for thorough comparison"""
    stacking_prediction_filename = 'cv_stacking_prediction.npy'
    test_stacking_prediction_filename = 'test_stacking_prediction.npy'

    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            cv_iterable: tp.Union[sklearn.model_selection.KFold, tp.Iterable],
            logger: Logger,
            metrics: tp.Dict[str, tp.Union[tp.Callable, str]],
            n_jobs=-1
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._cv_iterable = cv_iterable
        self.logger = logger
        self._metrics = metrics
        self._n_jobs = n_jobs

    def score_test(self, estimator, metrics, run, test_data=None):
        """
        Computes scores for test data and logs them to given run
        :param estimator: fitted estimator
        :param metrics: metrics to compute
        :param run: run to log into
        :param test_data: optional argument if one wants to pass augmented test dataset
        :return: None
        """
        if test_data is None:
            test_data = self.X_test
        test_scores = _score(estimator, test_data, self.y_test, metrics)
        run.log_values({key + '_test': value for key, value in test_scores.items()})

def score_cv(self, estimator, metrics, run):
    """
    computes scores on cross-validation
    :param estimator: estimator to fit
    :param metrics: metrics to compute
    :param run: run to log to
    :return: None
    """
    cross_val_results = cross_validate(
        estimator=estimator,
        X=self.X_train,
        y=self.y_train,
        cv=self._cv_iterable,
        scoring=metrics,
        n_jobs=self._n_jobs,
        return_train_score=False
    )

    for key, value in cross_val_results.items():
        if key.startswith('test_'):
            metric_name = key.split('_', maxsplit=1)[1]
            mean_score = np.mean(value)
            std_score = np.std(value)
            run.log_values(
                {
                    metric_name + '_mean': mean_score,
                    metric_name + '_std': std_score
                }
            )

def generate_stacking_predictions(self, estimator, run):
    """
    generates predictions over cross-validation folds, then saves them as artifacts
    returns fitted estimator for convinience and les train overhead
    :param estimator: estimator to use
    :param run: run to log to
    :return: estimator fitted on train, stacking cross-val predictions, stacking test predictions
    """
    # Choose the most informative prediction method
    if hasattr(estimator, "predict_proba"):  # choose the most informative method for stacking predictions
        method = "predict_proba"
    elif hasattr(estimator, "decision_function"):
        method = "decision_function"
    else:
        method = "predict"

    # Generate cross-validation predictions for stacking
    cross_val_stacking_prediction = cross_val_predict(
        estimator=estimator,
        X=self.X_train,
        y=self.y_train,
        cv=self._cv_iterable,
        method=method,
        n_jobs=self._n_jobs
    )

    # Log cross-validation predictions as artifact
    run.log_artifact(ExperimentHandler.stacking_prediction_filename,
                     lambda file: np.save(file, cross_val_stacking_prediction))

    # Fit estimator on full training data
    estimator.fit(self.X_train, self.y_train)

    # Generate predictions for test data
    test_stacking_prediction = getattr(estimator, method)(self.X_test)

    # Log test predictions as artifact
    run.log_artifact(ExperimentHandler.test_stacking_prediction_filename,
                     lambda file: np.save(file, test_stacking_prediction))

    return estimator, cross_val_stacking_prediction, test_stacking_prediction

    def get_metrics(self, estimator):
        """
        get callable metrics with estimator validation
        (e.g. estimator has predict_proba necessary for likelihood computation, etc)
        """
        return _check_multimetric_scoring(estimator, self._metrics)

    def run(self, estimator: sklearn.base.BaseEstimator, name=None):
        """
        perform run for given estimator
        :param estimator: estimator to use
        :param name: name of run for convinience and consitent logging
        :return: leaderboard with conducted run
        """
        metrics = self.get_metrics(estimator)
        with self.logger.run(name=name) as run:
            # compute predictions over cross-validation
            self.score_cv(estimator, metrics, run)
            fitted_on_train, _, _ = self.generate_stacking_predictions(estimator, run)
            self.score_test(fitted_on_train, metrics, run, test_data=self.X_test)
            return self.logger.leaderboard.loc[[run.name]]

    def get_stacking_predictions(self, run_names):
        """
        :param run_names: run names for which to extract stacking predictions for averaging and stacking
        :return: dataframe with predictions indexed by run names
        """
        train_dataframes = []
        test_dataframes = []
        for run_name in run_names:
            train_filename = self.logger.path / run_name / ExperimentHandler.stacking_prediction_filename
            train_dataframes.append(load_predictions_dataframe(filename=train_filename, column_prefix=run_name,
                                                               index=self.X_train.index))
            test_filename = self.logger.path / run_name / ExperimentHandler.test_stacking_prediction_filename
            test_dataframes.append(load_predictions_dataframe(filename=test_filename, column_prefix=run_name,
                                                              index=self.X_test.index))

        return pd.concat(train_dataframes, axis=1), pd.concat(test_dataframes, axis=1)