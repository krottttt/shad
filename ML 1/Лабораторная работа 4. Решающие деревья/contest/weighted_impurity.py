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
