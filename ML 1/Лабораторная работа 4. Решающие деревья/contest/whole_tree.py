import enum
import typing as tp

import numpy as np
from scipy.stats import mode


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
            self._class_proba = np.zeros(self._meta._n_classes)
            un = np.unique(y, return_counts=True)
            classes = un[0]
            counts = un[1]
            n = len(y)
            for i,c in enumerate(classes):
                self._class_proba[c] = counts[i]/n
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
        self._n_classes = np.unique(y).shape[0]
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
            y_pred = np.apply_along_axis(self.root.predict, 1, X)
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
            y_proba = np.apply_along_axis(self.root.predict_proba, 1,X )
            return y_proba

