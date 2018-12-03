# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:54:17 2018

@author: wli163
"""
import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
import numpy as np
from numba import njit
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import pdist, squareform

from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print
from pyod.models.loci import LOCI

contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points

# Generate sample data
X_train, y_train, X_test, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=2,
                  contamination=contamination,
                  random_state=42)

# train kNN detector
clf_name = 'LOCI'
clf = LOCI()

from loci.loci import run_loci
import matplotlib.pyplot as plt
data = X_test
loci_res = run_loci(data)
outlier_indices = loci_res.outlier_indices
print(outlier_indices)

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(data[outlier_indices, 0], data[outlier_indices, 1], c='r')
plt.show()
