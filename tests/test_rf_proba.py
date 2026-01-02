import numpy as np
from decision_tree.random_forest import RandomForestClassifier

def test_rf_predict_proba_sums_and_matches_predict():
    X = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.9],
        [0.9, 0.2],
    ])
    y = np.array([0, 1, 0, 1, 0, 1])

    rf = RandomForestClassifier(n_estimators=5, max_depth=3, seed=42)
    rf.fit(X, y)

    proba = rf.predict_proba(X)
    pred = rf.predict(X)

    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(pred, np.argmax(proba, axis=1))
