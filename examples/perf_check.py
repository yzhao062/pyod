# -*- coding: utf-8 -*-
"""Example of combining multiple base outlier scores. Four combination
frameworks are demonstrated:
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np

from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

if __name__ == "__main__":

    file_list = ['arrhythmia.csv', 'cardio.csv', 'ionosphere.csv',
                 'letter.csv', 'pima.csv']
    # Define data file and read X and y
    # Generate some data if the source data is missing

    for csv_file in file_list:

        try:
            data = np.genfromtxt(os.path.join('data', csv_file),
                                 delimiter=',', skip_header=1)

        except IOError:
            print('{data_file} does not exist. Use generated data'.format(
                data_file=csv_file))
            X, y = generate_data(train_only=True)  # load data
        else:
            X = data[:, :-1]
            y = data[:, -1].astype(int)

        clf = KNN() # the algorithm you want to check
        # clf = KNN_new()
        clf.fit(X) # fit model

        # print performance
        evaluate_print(csv_file, y, clf.decision_scores_)
