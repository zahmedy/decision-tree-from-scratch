import numpy as np

from decision_tree import gini, gini_gain


def test_gini_zero_for_pure_split():
    assert gini(np.array([1, 1, 1])) == 0.0


def test_gini_gain_prefers_informative_split():
    y_parent = np.array([0, 0, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1])

    assert gini_gain(y_parent, y_left, y_right) > 0
