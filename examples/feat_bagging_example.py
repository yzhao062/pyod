# %%
"""
Example of using Feature Bagging for outlier detection
"""
import os, sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from pyod.models.feat_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.base import clone
from pyod.utils.load_data import generate_data
from pyod.utils.utility import precision_n_scores
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

contamination = 0.1  # percentage of outliers
n_train = 100
n_test = 50

X_train, y_train, c_train, X_test, y_test, c_test = generate_data(
    n_train=n_train, n_test=n_test, contamination=contamination)

# clf = FeatureBagging(base_estimator=KNN(n_neighbors=6))
#clf = KNN()
clf = LOF()
#clf = LocalOutlierFactor()
clf.fit(X_train, y_train)
sc = clf.predict_proba(X_test)

#print(hasattr(clf, "predict_proba"))
# print(clf.predict_proba(X_test).shape)

# clf = KNN()
# clf.fit(X_train_)
# print(clf.predict_proba(X_test).shape)
#
check_estimator(clf)
#clf2 = clone(clf)

#clf = LogisticRegression()
#clf.fit(X_train_, y_train)
#check_estimator(clf)
#
# scores = clf.predict_proba(X_test)
