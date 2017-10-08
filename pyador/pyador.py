'''
Unsupervised anamoly detection toolbox in Python
'''

# Author: 'Yue Zhao' <yuezhao@cs.toronto.edu>
# License: MIT

import os
import errno

from sklearn.ensemble import IsolationForest

import pyador.local as const

from .util.data_prep import _missing_check
from .util.data_prep import _integrity_check
from .util.data_prep import _cat_to_num


class Pyador:
    def __init__(self, n=None, frac=None):

        self.n = n
        self.frac = frac
        self.X = None
        self.num_X = None
        self.le_dict = None
        self.y = None

        self._setup()
        self._arg_check()

    def fit_predict(self, X):

        self.X, self.num_X, self.le_dict = self._data_check_fix(X)

        # calculate the frac for the model digestion
        if self.frac is None:
            self.frac = float(self.n) / self.X.shape[0]

        if self.n is None:
            self.n = int(self.frac * self.X.shape[0])

        if self.X.shape[0] <= self.n <= 0:
            raise ValueError("n should be between 0 and the number of samples")

        self.y = self._fit()
        return self.y

    def _fit(self):

        clf = IsolationForest(contamination=self.frac)

        clf.fit(self.num_X)
        return clf.predict(self.num_X)

    def _data_check_fix(self, X):
        # check data type
        _integrity_check(X)

        X = _missing_check(X)
        X, num_X, le_dict = _cat_to_num(X)
        return X, num_X, le_dict

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
            print("anamoly fraction =", self.frac)
        if self.n is not None:
            print("anamoly number =", self.n)
        if self.y is not None:
            print("Identified %i potential anamolies" %len(self.y))

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
