"""
Locally Selective Combination of Parallel Outlier Ensembles (LSCP)
Adapted from the original implementation:
"""
# Author: Zain Nasrullah
# License: BSD 2 clause

import os
import sys
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

if __name__ == "__main__":

    from pyod.models.lscp import LSCP
    from pyod.models.lof import LOF
    from pyod.utils.utility import standardizer

    import scipy.io as scio
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score


    def load_data(filename):
        """
        load data
        :param filename:
        :return:
        """
        mat = scio.loadmat(filename)
        X_orig = mat['X']
        y_orig = mat['y'].ravel()
        return X_orig, y_orig


    X, y = load_data(r"data/cardio.mat")

    random_state = np.random.RandomState(0)

    el = []
    k_list = random_state.randint(5, 200, size=50).tolist()
    for k in k_list:
        el.append(LOF(k))

    # create the model
    clf = LSCP(el, random_state=random_state, local_region_size=100)

    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_train, X_test = standardizer(X_train, X_test)

    # fit and predict
    clf.fit(X_train)
    scores = clf.decision_function(X_test)
    print(roc_auc_score(y_test, scores))
