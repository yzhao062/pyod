# -*- coding: utf-8 -*-
"""Example of using Histogram- based outlier detection (HBOS) for
categorical real-world datasets
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
from urllib.request import urlopen
import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.hbos import HBOS

def breast_cancer_dataset():
    """
    Invoking this function will execute HBOS on Breast Cancer Dataset obtained online.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer" \
          ".data"
    raw_data = urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",", dtype=str)

    print("Dataset --BREAST CANCER--: \n{}\n".format(dataset))

    X, Y = dataset[:, range(dataset.shape[1] - 1)], \
             [1 if i == 'yes' else 0 for i in dataset[:, dataset.shape[1] - 1]]

    print("#Observations: {} , ... #Features: {}\n".format(dataset.shape[0], dataset.shape[1]))
    print("Outliers Ratio: {} %\n".format(np.unique(Y, return_counts=True)[1][1]
                                          * 100. / dataset.shape[0]))

     # train HBOS detector
    clf = HBOS(category='oneHot')
    clf.fit(X)

    print("Precision: {} %".format(
        np.sum(clf.labels_[np.where(clf.labels_ == Y)]) * 100. / np.sum(clf.labels_)))

def car_evaluation_dataset():
    """
    Invoking this function will execute HBOS on Car Evaluation Dataset obtained online.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    raw_data = urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",", dtype=str)

    print("Dataset --CAR EVALUATION--: \n{}\n".format(dataset))

    X, Y = dataset[:, range(dataset.shape[1] - 1)], \
           [1 if i in ('good', 'vgood') else 0 for i in dataset[:, dataset.shape[1] - 1]]

    print("#Observations: {} , ... #Features: {}\n".format(dataset.shape[0], dataset.shape[1]))
    print("Outliers Ratio: {} %\n".format(np.unique(Y, return_counts=True)[1][1] * 100. /
                                          dataset.shape[0]))

    # train HBOS detector
    clf = HBOS(category='frequency')
    clf.fit(X)

    print("Precision: {} %".format(
        np.sum(clf.labels_[np.where(clf.labels_ == Y)]) * 100. / np.sum(clf.labels_)))

def tic_tac_toe_dataset():
    """
    Invoking this function will execute HBOS on Tic Tac Toe Dataset obtained online.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
    raw_data = urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",", dtype=str)

    print("Dataset --TIC TAC TOE--: \n{}\n".format(dataset))

    X, Y = dataset[:, range(dataset.shape[1] - 1)], \
             [1 if i == 'negative' else 0 for i in dataset[:, dataset.shape[1] - 1]]

    print("#Observations: {} , ... #Features: {}\n".format(dataset.shape[0], dataset.shape[1]))
    print("Outliers Ratio: {} %\n".format(np.unique(Y, return_counts=True)[1][1]
                                          * 100. / dataset.shape[0]))

     # train HBOS detector
    clf = HBOS(category='label')
    clf.fit(X)

    print("Precision: {} %".format(
        np.sum(clf.labels_[np.where(clf.labels_ == Y)]) * 100. / np.sum(clf.labels_)))


if __name__ == "__main__":
    breast_cancer_dataset()
    #car_evaluation_dataset()
    #tic_tac_toe_dataset()
