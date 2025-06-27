import contextlib
import enum
import json
import os
import pathlib
import typing as tp
import uuid

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import mode
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.model_selection import cross_validate,cross_val_predict


class NodeType(enum.Enum):
    REGULAR = 1
    TERMINAL = 2



def gini(y: np.ndarray) -> float:
    """
    Computes Gini index for given set of labels
    :param y: labels
    :return: Gini impurity
    """
    set_y = set(y)
    n = len(y)
    gini_index = 1.0
    for label in set_y:
        count_label = np.sum(y == label)
        gini_index -= (count_label / n)**2
    return gini_index


def weighted_impurity(y_left: np.ndarray, y_right: np.ndarray) -> \
        tp.Tuple[float, float, float]:
    """
    Computes weighted impurity by averaging children impurities
    :param y_left: left  partition
    :param y_right: right partition
    :return: averaged impurity, left child impurity, right child impurity
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n = n_left + n_right
    left_impurity = gini(y_left)
    right_impurity = gini(y_right)
    y = np.concatenate([y_left, y_right], axis=0)
    weighted_impurity = n_left*left_impurity/n + n_right*right_impurity/n
    return weighted_impurity, left_impurity, right_impurity


def create_split(feature_values: np.ndarray, threshold: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    splits given 1-d array according to relation to threshold into two subarrays
    :param feature_values: feature values extracted from data
    :param threshold: value to compare with
    :return: two sets of indices
    """
    left_idx = np.where(feature_values <= threshold)
    right_idx = np.where(feature_values > threshold)
    return left_idx, right_idx


def _best_split(self, X: np.ndarray, y: np.ndarray):
    """
    finds best split
    :param X: Data, passed to node
    :param y: labels
    :return: best feature, best threshold, left child impurity, right child impurity
    """
    lowest_impurity = np.inf
    best_feature_id = None
    best_threshold = None
    lowest_left_child_impurity, lowest_right_child_impurity = None, None
    features = self._meta.rng.permutation(X.shape[1])
    for feature in features:
        current_feature_values = X[:, feature]
        thresholds = np.unique(current_feature_values)
        for threshold in thresholds[:-1]:
            # find indices for split with current threshold
            # current_weighted_impurity, current_left_impurity, current_right_impurity = ...  # calcualte impurity for given indices
            left_idx, right_idx = create_split(current_feature_values, threshold)
            current_weighted_impurity, current_left_impurity, current_right_impurity = weighted_impurity(y[left_idx], y[right_idx])
            if current_weighted_impurity < lowest_impurity:
                lowest_impurity = current_weighted_impurity
                best_feature_id = feature
                best_threshold = threshold
                lowest_left_child_impurity = current_left_impurity
                lowest_right_child_impurity = current_right_impurity

    return best_feature_id, best_threshold, lowest_left_child_impurity, lowest_right_child_impurity


class MyDecisionTreeNode:
    """
    Auxiliary class serving as representation of a decision tree node
    """

    def __init__(
            self,
            meta: 'MyDecisionTreeClassifier',
            depth,
            node_type: NodeType = NodeType.REGULAR,
            predicted_class: tp.Optional[tp.Union[int, str]] = None,
            left_subtree: tp.Optional['MyDecisionTreeNode'] = None,
            right_subtree: tp.Optional['MyDecisionTreeNode'] = None,
            feature_id: int = None,
            threshold: float = None,
            impurity: float = np.inf
    ):
        """

        :param meta: object, holding meta information about tree
        :param depth: depth of this node in a tree (is deduced on creation by depth of ancestor)
        :param node_type: 'regular' or 'terminal' depending on whether this node is a leaf node
        :param predicted_class: class label assigned to a terminal node
        :param feature_id: index if feature to split by
        :param
        """
        self._node_type = node_type
        self._meta = meta
        self._depth = depth
        self._predicted_class = predicted_class
        self._class_proba = None
        self._left_subtree = left_subtree
        self._right_subtree = right_subtree
        self._feature_id = feature_id
        self._threshold = threshold
        self._impurity = impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        finds best split
        :param X: Data, passed to node
        :param y: labels
        :return: best feature, best threshold, left child impurity, right child impurity
        """
        lowest_impurity = np.inf
        best_feature_id = None
        best_threshold = None
        lowest_left_child_impurity, lowest_right_child_impurity = None, None
        features = self._meta.rng.permutation(X.shape[1])
        for feature in features:
            current_feature_values = X[:, feature]
            thresholds = np.unique(current_feature_values)
            for threshold in thresholds[:-1]:
                # find indices for split with current threshold
                # current_weighted_impurity, current_left_impurity, current_right_impurity = ...  # calcualte impurity for given indices
                left_idx, right_idx = create_split(current_feature_values, threshold)
                current_weighted_impurity, current_left_impurity, current_right_impurity = weighted_impurity(
                    y[left_idx], y[right_idx])
                if current_weighted_impurity < lowest_impurity:
                    lowest_impurity = current_weighted_impurity
                    best_feature_id = feature
                    best_threshold = threshold
                    lowest_left_child_impurity = current_left_impurity
                    lowest_right_child_impurity = current_right_impurity

        return best_feature_id, best_threshold, lowest_left_child_impurity, lowest_right_child_impurity

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        recursively fits a node, providing it with predicted class or split condition
        :param X: Data
        :param y: labels
        :return: fitted node
        """
        if (
            self._depth >= self._meta.max_depth
                or len(y)<self._meta.min_samples_split
                or self._impurity == 0
                or gini(y) == 0
                or len(np.unique(X,axis=0)) <= 1
        ):
            self._node_type = NodeType.TERMINAL
            self._predicted_class = mode(y)[0]
            self._class_proba = np.bincount(y.astype(int),minlength = 2)/y.shape[0]
            return self

        self._feature_id, self._threshold, left_imp, right_imp = self._best_split(X, y)
        left_idx, right_idx = create_split(X[:, self._feature_id], self._threshold)
        self._left_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth+1,
            impurity=left_imp
        ).fit(
            X[left_idx],y[left_idx]
        )
        self._right_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth+1,
            impurity=right_imp
        ).fit(
            X[right_idx],y[right_idx]
        )
        return self

    def predict(self, x: np.ndarray):
        """
        Predicts class for a single object
        :param x: object of shape (n_features, )
        :return: class assigned to object
        """

        if self._node_type is NodeType.TERMINAL:
            return self._predicted_class
        if x[self._feature_id] <= self._threshold:
            return self._left_subtree.predict(x)
        else:
            return self._right_subtree.predict(x)

    def predict_proba(self, x: np.ndarray):
        """
        Predicts probability for a single object
        :param x: object of shape (n_features, )
        :return: vector of probabilities assigned to object
        """
        if self._node_type is NodeType.TERMINAL:
            return self._class_proba
        if x[self._feature_id] <= self._threshold:
            return self._left_subtree.predict_proba(x)
        else:
            return self._right_subtree.predict_proba(x)


class MyDecisionTreeClassifier:
    """
    Class analogous to sklearn implementation of decision tree classifier with Gini impurity criterion,
    named in a manner avoiding collisions
    """

    def __init__(
            self,
            max_depth: tp.Optional[int] = None,
            min_samples_split: tp.Optional[int] = 2,
            seed: int = 0
    ):
        """
        :param max_depth: maximal depth of tree, prevents overfitting
        :param min_samples_split: minimal amount of samples for node to be a splitter node
        :param seed: seed for RNG, enables reproducibility
        """
        self.root = MyDecisionTreeNode(self, 1)
        self._is_trained = False
        self.max_depth = max_depth or np.inf
        self.min_samples_split = min_samples_split or 2
        self.rng = np.random.default_rng(seed)
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        starts recursive process of node criterion fitting from the root
        :param X: Data
        :param y: labels
        :return: fitted self
        """
        # self.labels = np.unique(y)
        # self._n_classes = self.labels.shape[0]
        self.root.fit(X, y)
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class for a sequence of objects
        :param x: Data
        :return: classes assigned to each object
        """
        if not self._is_trained:
            raise RuntimeError('predict call on untrained model')
        else:
            y_pred = np.zeros(X.shape[0])
            for i,x in enumerate(X):
                y_pred[i] = self.root.predict(x)
            return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class for a sequence of objects
        :param x: Data
        :return: probabilities of all classes for each object
        """
        if not self._is_trained:
            raise RuntimeError('predict call on untrained model')
        else:
            y_proba = list()
            for i, x in enumerate(X):
                y_proba.append(self.root.predict_proba(x))
            y_proba = np.array(y_proba)
            return y_proba


class MyRandomForestClassifier:
    """
    Data-diverse ensemble of tree calssifiers
    """
    big_number = 1 << 32

    def __init__(
            self,
            n_estimators: int,
            max_depth: tp.Optional[int] = None,
            min_samples_split: tp.Optional[int] = 2,
            seed: int = 0
    ):
        """
        :param n_estimators: number of trees in forest
        :param max_depth: maximal depth of tree, prevents overfitting
        :param min_samples_split: minimal amount of samples for node to be a splitter node
        :param seed: seed for RNG, enables reproducibility
        """
        self._n_classes = 0
        self._is_trained = False
        self.rng = np.random.default_rng(seed)
        self.n_estimators = n_estimators
        self.estimators = [
            MyDecisionTreeClassifier(max_depth, min_samples_split, seed=seed) for
            seed in self.rng.choice(max(MyRandomForestClassifier.big_number, n_estimators), size=(n_estimators,),
                                    replace=False)]

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """
        returns bootstrapped sample from X of equal size
        :param X: objects collection to sample from
        :param y: corresponding labels
        :return:
        """
        n = y.shape[0]
        indices = np.random.randint(0, n, size=n)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        fits each estimator of the ensemble on the bootstrapped data sample
        :param X: Data
        :param y: labels
        :return: fitted self
        """
        self.labels = np.unique(y)
        self._n_classes = self.labels.shape[0]
        for estimator in self.estimators:
            # estimator._n_classes = self._n_classes
            X_b, y_b = self._bootstrap_sample(X, y)
            estimator.fit(X_b, y_b)
        self._is_trained = True
        return self

    def predict_proba(self, X: np.ndarray):
        """
        predict probability of each class by averaging over all base estimators
        :param X: Data
        :return: array of probabilities
        """
        if self.n_estimators == 0:
            raise RuntimeError('predict call on empty ensemble')
        if not self._is_trained:
            raise RuntimeError('predict call on untrained model')
        # if self._n_classes <= 1:
        #     raise RuntimeError('zero or single class')
        else:
            norm = 0
            probas = np.zeros((X.shape[0], self._n_classes))
            # probas = list()
            for estimator in self.estimators:
                if estimator._is_trained:
                    norm+=1
                    for i,p in enumerate(estimator.predict_proba(X)):
                        probas[i] += p
                        # probas += estimator.predict_proba(X)
            #for i,p in enumerate(probas):
            #    norm = np.sum(p)
            #    if norm == 0:
            #        norm = 1
            #    probas[i] /= norm
            for i, p in enumerate(probas):
                norm = np.sum(p)
                if norm == 0:
                    norm = 1
                probas[i] = np.divide(p,norm)
            # probas = np.divide(probas,float(norm))
            return probas
            # else:
            #     raise RuntimeError('predict call on empty dataset')

    def predict(self, X):
        """
        predict class for each object
        :param X: Data
        :return: array of class labels
        """
        if self.n_estimators == 0:
            raise RuntimeError('predict call on empty ensemble')
        if not self._is_trained:
            raise RuntimeError('predict call on untrained model')
        # if self._n_classes <= 1:
        #     raise RuntimeError('zero or single class')
        else:
            if X.shape[0] > 0:
                if len(X.shape) == 1:
                    n_objects = 1
                else:
                    n_objects = X.shape[0]
                y_pred = np.ones(n_objects)*2
                probas = self.predict_proba(X)
                if probas is None:
                    return np.zeros(n_objects)
                else:
                    for i,p in enumerate(probas):
                        arg = np.argmax(p)
                        y_pred[i] = arg
                    return y_pred
                    # return np.apply_along_axis(np.argmax, 1, probas)
            else:
                raise RuntimeError('predict call on empty dataset')


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
