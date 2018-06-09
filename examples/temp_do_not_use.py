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

from pyod.models.pca import PCA
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.data import _generate_data

import matplotlib.pyplot as plt 

if __name__ == "__main__":
    
    X,y = _generate_data(100, 50, 2, 42)
    
    fig = plt.figure(figsize=(12, 10))
    
    plt.scatter(X[:,0],X[:,1], c=y)

