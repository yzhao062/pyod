'''
Unsupervised anamoly detection toolbox in Python
'''

# Author: 'Yue Zhao' <yuezhao@cs.toronto.edu>
# License: MIT

import os, errno
import pyador.local as const
from .util.data_prep import miss_check


class Pyador:
    def __init__(self):
        self._setup()

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
