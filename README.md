# decision-tree-from-scratch
Learning decision-tree internals by creating it from scratch  

```
decision-tree-from-scratch/
  README.md
  pyproject.toml            # optional, but nice
  src/
    tree/
      __init__.py
      criteria.py           # gini/entropy, info gain
      node.py               # Node dataclass
      split.py              # best split search
      decision_tree.py      # fit/predict
      utils.py              # small helpers
  tests/
    test_criteria.py
    test_tree_small.py
  examples/
    train_iris_like.py      # small demo using sklearn datasets (only for data)
```