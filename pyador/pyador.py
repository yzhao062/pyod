'''
Unsupervised anamoly detection toolbox in Python
'''

# Author: 'Yue Zhao' <yuezhao@cs.toronto.edu>
# License: MIT

import os, errno
import pyador.local as const
from .util.data_prep import _missing_check
from .util.data_prep import _integrity_check
from .util.data_prep import _cat_to_num


class Pyador:
    def __init__(self, X, n=None, frac=None):
        self._setup()
        self._arg_check(X, n, frac)
        self.X = _missing_check(self.X)
        self.X, self.le_dict = _cat_to_num(self.X)

    def _arg_check(self, X, n, frac):
        # check data type
        if _integrity_check(X):
            self.X = X

        # check the validity of the arguments
        if n is not None:
            self.n = n

        if frac is not None:
            if not 0 < frac < 1:
                raise ValueError("frac should be between 0 to 1")

            self.frac = frac

        if hasattr(self, "n") and hasattr(self, "frac"):
            raise ValueError("n and frac cannot be used at the same time. " \
                             "Use either n or frac instead")

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
