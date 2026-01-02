import numpy as np
from decision_tree.decision_tree import DecisionTreeClassifier

def test_tree_predict_proba_sums_to_one():
    X = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 0, 1])

    tree = DecisionTreeClassifier(max_depth=2, min_samples_split=2, seed=42)
    tree.fit(X, y)

    proba = tree.predict_proba(X)

    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
