import numpy as np 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from decision_tree.random_forest import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=8, 
    min_samples_split=2,
    seed=42
)
rf.fit(X_train, y_train)

test_acc = (rf.predict(X_test) == y_test).mean()

print("OOB score:", rf.oob_score_)
print("Test accuracy:", test_acc)
