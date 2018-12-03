# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from sklearn.utils import check_X_y
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from scipy.io import loadmat

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print
from pyod.utils.utility import standardizer


if __name__ == "__main__":

    # Define data file and read X and y
    # Generate some data if the source data is missing
    mat_file = 'cardio.mat'
    try:
        mat = loadmat(os.path.join('data', mat_file))

    except TypeError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    except IOError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    else:
        X = mat['X']
        y = mat['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.4,
                                                        random_state=1)
#    X_train_norm, X_test_norm = X_train, X_test
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    estimator_list = []
    normalization_list = []
    
    # predefined range of k
    k_range = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 
               60, 70, 80, 90, 100, 150, 200, 250]
    # validate the value of k
    k_range = [k for k in k_range if k < X.shape[0]]
    
    for k in k_range:
        estimator_list.append(KNN(n_neighbors=k))
        estimator_list.append(LOF(n_neighbors=k))
        normalization_list.append(True)
        normalization_list.append(True)
    
    n_bins_range = [3, 5, 7, 9, 12, 15, 20, 25, 30, 50]
    for n_bins in n_bins_range:
        estimator_list.append(HBOS(n_bins=n_bins))
        normalization_list.append(False)
        
    # predefined range of nu for one-class svm
    nu_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    for nu in nu_range:
        estimator_list.append(OCSVM(nu=nu))
        normalization_list.append(True)
    
    # predefined range for number of estimators in isolation forests
    n_range = [10, 20, 50, 70, 100, 150, 200, 250]
    for n in n_range:
        estimator_list.append(IForest(n_estimators=n))
        normalization_list.append(False)
    
    X_train_add = np.zeros([X_train.shape[0], len(estimator_list)])
    X_test_add = np.zeros([X_test.shape[0], len(estimator_list)])
    
    # fit the model
    for index, estimator in enumerate(estimator_list):
        estimator.fit(X_train_norm)
        X_train_add[:, index] =  estimator.decision_scores_
        X_test_add[:, index] =  estimator.decision_function(X_test_norm)
        
    # prepare the new feature space
    
    X_train_new = np.concatenate((X_train, X_train_add), axis=1)
    X_test_new = np.concatenate((X_test, X_test_add), axis=1)  
    
    clf = XGBClassifier()
    clf.fit(X_train_new, y_train)
    y_test_scores = clf.predict_proba(X_test_new)  # outlier scores
    
    evaluate_print('XGBOD', y_test, y_test_scores[:, 1])
    
    
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    y_test_scores_orig = clf.predict_proba(X_test)  # outlier scores
    
    evaluate_print('old', y_test, y_test_scores_orig[:, 1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    