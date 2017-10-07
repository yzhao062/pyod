'''
Unsupervised anamoly detection toolbox in Python
'''

# Author: 'Yue Zhao' <yuezhao@cs.toronto.edu>
# License: MIT

import os, errno
import pyador.local as const
from .util.data_prep import miss_check
from .util.data_prep import integrity_check


class Pyador:
    def __init__(self, X, n=None, frac=None):
        self._setup()

        # check data type
        integrity_check(X)

        if n is not None:
            self.n = n
        if frac is not None:
            self.frac = frac

        if hasattr(self, "n") and hasattr(self, "frac"):
            raise ValueError("n and frac cannot be used at the same time. " \
                             "Use either n or frac instead")

        self.n = n
        self.X = X

    def debug(self):
        print(self.X.shape)
        if hasattr(self, "frac"):
            print("frac =", self.frac)
        if hasattr(self, "n"):
            print("n =", self.n)

    def fit(self):
        pass

    def _setup(self):
        ''' perform necessary setup steps

        :return:
        '''
        # create output folder
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
