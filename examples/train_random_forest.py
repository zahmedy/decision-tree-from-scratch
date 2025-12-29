import numpy as np
from decision_tree.random_forest import RandomForestClassifier

X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [8, 80],
    [9, 90],
    [10, 100]
])
y = np.array([0, 0, 0, 1, 1, 1])

rf = RandomForestClassifier(n_estimators=50, max_depth=3)
rf.fit(X, y)

print(rf.predict(X))