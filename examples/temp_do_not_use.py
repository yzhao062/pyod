# -*- coding: utf-8 -*-
"""Example of combining multiple base outlier scores. Four combination
frameworks are demonstrated:

1. Average: take the average of all base detectors
2. maximization : take the maximum score across all detectors as the score
3. Average of Maximum (AOM)
4. Maximum of Average (MOA)

"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from scipy.stats import rankdata

from pyod.models.pca import PCA
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

import time


import matplotlib.pyplot as plt 

if __name__ == "__main__":
    
    X,y, Xt,yt = generate_data(10000, 5000, random_state=42)
    
    X1 = X[:,0]
    Xt1 = Xt[:,0]
    
    ranks = np.zeros([Xt.shape[0], ])
    start_time = time.clock()


    for i in range(Xt1.shape[0]):
        train_scores_i = np.append(X1.reshape(-1, 1), Xt1[i])

        ranks[i] = rankdata(train_scores_i)[-1]
    
    print("--- %s seconds ---" % (time.clock() - start_time))
    start_time = time.clock()
    ranks1 = np.zeros([Xt.shape[0], ])
    X1_sorted = np.sort(X1)
#    for i in range(Xt1.shape[0]):
#    
#        ranks[i] = np.searchsorted(X1_sorted)
    ranks1 = np.searchsorted(X1_sorted, Xt1)
    
    print("--- %s seconds ---" % (time.clock() - start_time))
    

    


