# -*- coding: utf-8 -*-
"""Example of using and visualizing ``generate_data_clusters`` function
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.utils.data import generate_data_clusters
from pyod.utils.example import data_visualize

if __name__ == "__main__":
    contamination = 0.05  # percentage of outliers
    n_samples = 500  # number of sample points
    test_size = None  # None means no test split is wanted

    # Generate sample data in clusters
    x, y = generate_data_clusters(n_samples=n_samples,
                                  test_size=test_size,
                                  n_clusters=3,
                                  n_features=2,
                                  contamination=contamination,
                                  size='different',
                                  density='different',
                                  dist=0.2,
                                  random_state=42,
                                  return_in_clusters=True)

    # visualize the results
    data_visualize(x, y, show_figure=True, save_figure=False)
