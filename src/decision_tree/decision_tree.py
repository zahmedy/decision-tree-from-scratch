import numpy as np
from decision_tree.node import Node
from decision_tree.split import best_split
from decision_tree.criteria import gini


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _majority_class(self, y: np.ndarray) -> int:
        return int(np.bincount(y).argmax())

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # 1) stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split or gini(y) == 0.0:
            return Node(is_leaf=True, prediction=self._majority_class(y))

        # 2) find best split
        feature, threshold, gain, left_idx, right_idx = best_split(X, y)

        # 3) if no useful split
        if feature is None or gain <= 1e-12:
            return Node(is_leaf=True, prediction=self._majority_class(y))

        # 4) split data and recurse
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        # 5) return internal node
        return Node(
            is_leaf=False,
            feature_index=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf:
            return node.prediction

        # decide direction
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
