import numpy as np
from decision_tree.decision_tree import DecisionTreeClassifier



class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees = []
        self.rng = np.random.default_rng(seed)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        idx = self.rng.integers(0, n, size=n)
        return X[idx], y[idx]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        for _ in range(self.n_estimators):
            tree_seed = int(self.rng.integers(0, 1_000_000_000))
            Xb, yb = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                seed=tree_seed,
                max_features="sqrt"
                )
            tree.fit(Xb, yb)
            self.trees.append(tree)

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        y_hat = (all_preds.mean(axis=0) >= 0.5).astype(int)
        return y_hat


    