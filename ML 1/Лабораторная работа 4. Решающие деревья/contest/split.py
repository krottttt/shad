import numpy as np
import typing as tp


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