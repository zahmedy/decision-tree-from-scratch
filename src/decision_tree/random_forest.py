import numpy as np
from decision_tree.decision_tree import DecisionTreeClassifier



class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, max_features=None, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.rng = np.random.default_rng(seed)
        self.oob_masks = []
        self.oob_score_ = None

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        idx = self.rng.integers(0, n, size=n)
        return X[idx], y[idx], idx
    
    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        n = X.shape[0]

        vote_sum = np.zeros(n, dtype=float)
        votes_count = np.zeros(n, dtype=int)
    
        for tree, oob_mask in zip(self.trees, self.oob_masks):
            oob_idx = np.where(oob_mask)[0]
            if oob_idx.size == 0:
                continue
            
            preds = tree.predict(X[oob_idx])
            vote_sum[oob_idx] += preds
            votes_count[oob_idx] += 1

        valid = votes_count > 0
        oob_pred = np.zeros(n, dtype=int)
        oob_pred[valid] = (vote_sum[valid] / votes_count[valid] >= 0.5).astype(int)

        return (oob_pred[valid] == y[valid]).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        
        self.trees = []
        for _ in range(self.n_estimators):
            tree_seed = int(self.rng.integers(0, 1_000_000_000))
            Xb, yb, idx = self._bootstrap_sample(X, y)
            
            n = X.shape[0]
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[idx] = False

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                seed=tree_seed,
                max_features=self.max_features
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)
            self.oob_masks.append(oob_mask)
            self.oob_score_ = self._compute_oob_score(X, y)

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        y_hat = (all_preds.mean(axis=0) >= 0.5).astype(int)
        return y_hat


    