# -*- coding: utf-8 -*-
"""Example of using and visualizing ``generate_data_categorical`` function.
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.utils.data import generate_data_categorical

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers

    # Generate sample data in clusters
    X_train, X_test, y_train, y_test = generate_data_categorical \
        (n_train=200, n_test=50,
         n_category_in=8, n_category_out=5,
         n_informative=1, n_features=1,
         contamination=contamination,
         shuffle=True, random_state=42)

    # note that visalizing it can only be in 1 dimension!
    cats = list(np.ravel(X_train))
    labels = list(y_train)
    fig, axs = plt.subplots(1, 2)
    axs[0].bar(cats, labels)
    axs[1].plot(cats, labels)
    plt.title('Synthetic Categorical Train Data')
    plt.show()

    cats = list(np.ravel(X_test))
    labels = list(y_test)
    fig, axs = plt.subplots(1, 2)
    axs[0].bar(cats, labels)
    axs[1].plot(cats, labels)
    plt.title('Synthetic Categorical Test Data')
    plt.show()
