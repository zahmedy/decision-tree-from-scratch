# Decision Tree from Scratch
Readable tree learning experiments: a minimal decision tree classifier, a bootstrap-averaged random forest with OOB scoring, and small examples that keep the math visible.

## What’s inside
- Gini impurity and information gain in `src/decision_tree/criteria.py`
- Exhaustive split search in `src/decision_tree/split.py`
- Minimal `DecisionTreeClassifier` in `src/decision_tree/decision_tree.py`
- Bootstrap ensemble with out-of-bag score in `src/decision_tree/random_forest.py`
- Small, focused tests in `tests/`
- Examples: `examples/train_iris_like.py` (toy tree) and `examples/train_random_forest.py` / `examples/run_rf_oob.py` (forest + OOB)

## Install
```bash
pip install -e .
```

## Quickstart
Decision tree:
```python
import numpy as np
from decision_tree import DecisionTreeClassifier

X = np.array([[2], [4], [6], [8]], dtype=float)
y = np.array([0, 0, 1, 1], dtype=int)

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2).fit(X, y)
print("Root feature:", clf.root.feature_index)
print("Root threshold:", clf.root.threshold)
print("Preds:", clf.predict(X))
```

Random forest with out-of-bag score:
```python
import numpy as np
from decision_tree.random_forest import RandomForestClassifier

X = np.array([[1, 10], [2, 20], [3, 30], [8, 80], [9, 90], [10, 100]], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1])

rf = RandomForestClassifier(n_estimators=50, max_depth=3, seed=42).fit(X, y)
print("Preds:", rf.predict(X))
print("OOB score:", rf.oob_score_)
```

## Run tests
```bash
pytest
```

## Project layout
```
decision-tree-from-scratch/
├─ src/decision_tree/
│  ├─ __init__.py
│  ├─ criteria.py
│  ├─ decision_tree.py
│  ├─ node.py
│  ├─ random_forest.py
│  └─ split.py
├─ examples/
│  ├─ train_iris_like.py
│  ├─ train_random_forest.py
│  └─ run_rf_oob.py
├─ tests/
│  ├─ test_criteria.py
│  └─ test_tree_small.py
└─ pyproject.toml
```
