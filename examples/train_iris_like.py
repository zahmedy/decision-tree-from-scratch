import numpy as np
from decision_tree import DecisionTreeClassifier

X = np.array([[2], [4], [6], [8]], dtype=float)
y = np.array([0,0,1,1], dtype=int)

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
clf.fit(X, y)

print(f"Root feature: {clf.root.feature_index}")
print(f"Root threshold: {clf.root.threshold}")
print(f"Preds: {clf.predict(X)}")
