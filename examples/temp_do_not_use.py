# -*- coding: utf-8 -*-
"""Example of using Cluster-based Local Outlier Factor (CBLOF) for outlier
detection
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

from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyod.models.cblof import CBLOF
from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print
from pyod.utils.utility import standardizer

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras import regularizers

# Loads the training and test data sets (ignoring class labels)
#(x_train, _), (x_test, _) = mnist.load_data()
contamination = 0.1  # percentage of outliers
n_train = 50000  # number of training points
n_test = 5000  # number of testing points

# Generate sample data
X_train, y_train, X_test, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=400,
                  contamination=contamination,
                  random_state=42)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

#%%
# input dimension = 128
input_dim = X_train.shape[1]
encoding_dim = 32
epochs = 10
batch_size = 32
dropout_rate = 0.2

compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential()

#autoencoder.add(
#    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
#)
#autoencoder.add(
#    Dense(input_dim, activation='sigmoid')
#)

autoencoder.add(Dense(64, activation='relu', input_shape=(input_dim,)))
autoencoder.add(Dropout(dropout_rate))
#autoencoder.add(Dense(64, activation='relu'))
#autoencoder.add(Dropout(dropout_rate))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dropout(dropout_rate))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dropout(dropout_rate))
#autoencoder.add(Dense(64, activation='relu'))
#autoencoder.add(Dropout(dropout_rate))
autoencoder.add(Dense(64, activation='relu'))
autoencoder.add(Dropout(dropout_rate))
autoencoder.add(Dense(input_dim, activation='sigmoid'))
autoencoder.compile(loss=keras.losses.mean_squared_error, optimizer='adam')

autoencoder.summary()

history = autoencoder.fit(X_train_norm, X_train_norm,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=True,
                         validation_split=0.1,
                         verbose=1).history
#%%
from sklearn.metrics.pairwise import euclidean_distances
autoencoder.summary()
pred_train = autoencoder.predict(X_train_norm)
pred_test = autoencoder.predict(X_test_norm)
#%%
from pyod.utils.stat_models import pairwise_distances_no_broadcast

#error_train = euclidean_distances(X_train_norm, pred_train)
#error_test = cdist(X_test_norm, pred_test, metric='euclidean')

train_error = pairwise_distances_no_broadcast(X_train_norm, pred_train)
test_error = pairwise_distances_no_broadcast(X_test_norm, pred_test)

#%%
from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyod.models.cblof import CBLOF
from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print
from pyod.utils.utility import standardizer

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras import regularizers
from pyod.models.auto_encoder import AutoEncoder
contamination = 0.1  # percentage of outliers
n_train = 50000  # number of training points
n_test = 5000  # number of testing points

# Generate sample data
X_train, y_train, X_test, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=400,
                  contamination=contamination,
                  random_state=42)
    
clf = AutoEncoder(epochs=20)
clf.fit(X_train)
#%%
train_scores = clf.decision_scores_
test_scores = clf.decision_function(X_test)

evaluate_print("AE", y_train, train_scores)
evaluate_print("AE", y_test, test_scores)
















