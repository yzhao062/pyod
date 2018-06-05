# -*- coding: utf-8 -*-
"""
Example of using kNN for outlier detection
"""
from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.pca import PCA
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

pca = decomposition.PCA()

x = np.array([[0.387, 4878, 5.42, 1.21],
              [0.723, 12104, 5.25, 2.22],
              [1, 12756, 5.52, 3.45],
              [1.524, 6787, 3.94, 2.1], ])

x_norm = StandardScaler().fit_transform(x)

# train PCA detector
clf_name = 'PCA'
clf = PCA()
clf.fit(x_norm)

###################################################################
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from pyod.utils.utility import get_label_n
# import numpy as np
# print('\nThe number of outliers is', np.count_nonzero(y_train))
#
# for n in range(15, 25):
#     y_pred = get_label_n(y_train, y_train_scores, n=n)
#     print('n:', n,
#           'precision:', precision_score(y_train, y_pred),
#           'recall:', recall_score(y_train, y_pred))
###################################################################

# %%
#import numpy as np
#from sklearn import decomposition
#from sklearn.preprocessing import StandardScaler
#
#pca = decomposition.PCA()
#
#x = np.array([[0.387, 4878, 5.42, 1.21],
#              [0.723, 12104, 5.25, 2.22],
#              [1, 12756, 5.52, 3.45],
#              [1.524, 6787, 3.94, 2.1], ])
#
#x_norm = StandardScaler().fit_transform(x)
#res = pca.fit_transform(x_norm)
#rev = pca.inverse_transform(res)
