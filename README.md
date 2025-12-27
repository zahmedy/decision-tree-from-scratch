# Decision Tree from Scratch
Tiny, readable decision tree classifier that shows each moving part: impurity, split search, nodes, and recursion.

## What’s inside
- Gini impurity and information gain in `src/decision_tree/criteria.py`
- Exhaustive split search in `src/decision_tree/split.py`
- Minimal `DecisionTreeClassifier` in `src/decision_tree/decision_tree.py`
- Small, focused tests in `tests/`
- Example script in `examples/train_iris_like.py`

## Install
```bash
pip install -e .
```

## Quickstart
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
│  └─ split.py
├─ examples/
│  └─ train_iris_like.py
├─ tests/
│  ├─ test_criteria.py
│  └─ test_tree_small.py
└─ pyproject.toml
```
