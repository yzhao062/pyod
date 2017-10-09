'''
Unsupervised anomaly detection toolbox in Python
'''

# Author: 'Yue Zhao' <yuezhao@cs.toronto.edu>
# License: MIT

import os
import errno

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

import pyador.local as const

from .util.data_prep import missing_check
from .util.data_prep import integrity_check
from .util.data_prep import cat_to_num


class Pyador:
    def __init__(self, n=None, frac=None):

        self.n = n
        self.frac = frac

        self.X = None
        self.num_X = None
        self.num_vars = None
        self.le_dict = None

        self.y_pred = None
        self.X_train = None
        self.Y_train = None

        self.clf = None

        self._setup()
        self._arg_check()

    def fit(self, X):

        para = self._data_check_fix(X)
        self.X = para["df"]
        self.num_X = para["num_df"]
        self.num_vars = para["num_vars"]
        self.le_dict = para["le_dict"]

        # calculate the frac for the model digestion
        if self.frac is None:
            self.frac = float(self.n) / self.X.shape[0]

        if self.n is None:
            self.n = int(self.frac * self.X.shape[0])

        if self.X.shape[0] <= self.n <= 0:
            raise ValueError("n should be between 0 and the number of samples")

        self.y_pred = self._predict_self()
        self.X_train, self.Y_train = self._build_label()
        self._train_clf()

        return self.y_pred, self.X_train, self.Y_train

    def _build_label(self):
        ''' merge potential anomalies with normal data

        anomaly: pred_target = -1
        normal data: pred_target = 1

        :return:
        '''

        anomaly_idx = np.where(self.y_pred == -1)
        normal_idx = np.where(self.y_pred == 1)

        anomaly_df = (self.num_X.loc[anomaly_idx]).copy(deep=True)
        normal_df = (self.num_X.loc[normal_idx]).sample(n=self.n).copy(
            deep=True)

        # generate the training df
        X_train = pd.concat([anomaly_df, normal_df], axis=0)

        anomaly_df["pred_target"] = -1
        normal_df["pred_target"] = 1

        Y_train = pd.concat(
            [anomaly_df["pred_target"], normal_df["pred_target"]], axis=0)

        # flatten as ndarray
        Y_train = Y_train.values.ravel()

        return X_train, Y_train

    def _train_clf(self):

        clf = RandomForestClassifier()
        clf.fit(self.X_train, self.Y_train)

        self.clf = clf
        accu_list = cross_val_score(clf, self.X_train, self.Y_train,
                                    scoring='accuracy', cv=10)

        print('10 fold cv is', accu_list.mean())

    def _predict_self(self):

        clf = IsolationForest(contamination=self.frac)

        clf.fit(self.num_X)

        return clf.predict(self.num_X)

    def predict(self, X_test):
        pass

    def _data_check_fix(self, X):
        # check data type
        integrity_check(X)

        X = missing_check(X)

        return cat_to_num(X)

    def _arg_check(self):

        # check the validity of the arguments
        if self.n is not None:
            if not self.n >= 0:
                raise ValueError("frac should be between 0 to 1")

        if self.frac is not None:
            if not 0 <= self.frac <= 1:
                raise ValueError("frac should be between 0 to 1")

        if self.n is not None and self.frac is not None:
            raise ValueError("n and frac cannot be used at the same time. " \
                             "Use either n or frac instead")

    def _debug(self):

        if self.frac is not None:
            print("anomaly fraction =", self.frac)
        if self.n is not None:
            print("anomaly number =", self.n)
        if self.y_pred is not None:
            print("Identified %i potential anomalies" % len(self.y))

    def _setup(self):
        ''' perform necessary setup steps

        :return:
        '''
        # create output folder
        # TODO: design the structure of the output
        output_dir = const.OUTPUT

        try:
            os.makedirs(output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @property
    def get_visuals(self):
        if hasattr(self, '_visual'):
            pass
        else:
            raise AttributeError(
                "object has no visuals as it is not fitted yet")
