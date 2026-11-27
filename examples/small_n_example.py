"""Example of using SmallN detector for scoring outliers
against a small reference set (n=3-10).
"""

import numpy as np

from pyod.models.small_n import SmallN

X_train = np.array([[1.0, 2.1], [1.1, 1.9], [0.9, 2.0]])
X_test = np.array([[1.0, 2.0], [8.0, 8.0]])

clf = SmallN(contamination=0.3)
clf.fit(X_train)

print("Decision scores on training data:", clf.decision_scores_)
print("Predicted labels on test data:", clf.predict(X_test))
