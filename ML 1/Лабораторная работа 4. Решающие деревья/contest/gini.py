import numpy as np

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