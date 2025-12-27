import numpy as np

from decision_tree import DecisionTreeClassifier


def test_predict_small_dataset():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2).fit(X, y)

    preds = clf.predict(X)
    np.testing.assert_array_equal(preds, y)

    # Root should split on the single feature
    assert clf.root.feature_index == 0
