import numpy as np

from decision_tree import DecisionTreeClassifier


def test_proba_small_dataset():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2).fit(X, y)

    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    assert proba.sum() == 1