# -*- coding: utf-8 -*-
"""Example of using Histogram- based outlier detection (HBOS) for
outlier detection
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from sklearn.cluster import MiniBatchKMeans
from pyod.utils.data import generate_data
from pyod.models.cblof import CBLOF
import numpy as np

#%%
x = [[0.30244003], 
     [0.01218177], 
     [-0.50835109], [-0.36951435], [0.97274482],
     [-0.68325119], [0.0], [0.0], [0.08], [0.0], [0.0], [0.0], [0.0], [0.0],
     [0.09], [0.0], [0.0], [0.0], [0.0], [0.0], [-20.29518778], [0.0], [0.0],
     [0.0], [0.0], [0.0], [0.0], [8.38548823], [0.0], [0.0]]

clf = MiniBatchKMeans(n_clusters=15)
clf.fit(x)
#test = generate_data(train_only=True)
#clf_name = 'CBLOF'
#clf = CBLOF(alpha=0.1, n_clusters=15, beta=10, check_estimator=False)
#try:
#    clf.fit(x)
#except Exception as ex:
#    print(str(ex))
#print("\n Cluster centers: " + str(clf.cluster_centers_))
#print("\n Cluster sizes: " + str(clf.cluster_sizes_))
#print('\n Supposed to be the cluster size: ' + str(
#    np.bincount(clf.cluster_labels_, minlength=15)))
#print("\n Large clusters: " + str(clf.large_cluster_labels_))
#print("\n Small clusters: " + str(clf.small_cluster_labels_))
