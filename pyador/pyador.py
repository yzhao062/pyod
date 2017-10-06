'''
Unsupervised anamoly detection toolbox in Python
'''

# Author: 'Yue Zhao' <yuezhao@cs.toronto.edu>
# License: MIT

from .util.data_prep import miss_check


class Pyador:
    def __init__(self):
        pass

    def fit(self):
        pass

    @property
    def get_visuals(self):
        if hasattr(self, '_visual'):
            pass
        else:
            raise AttributeError(
                "object has no visuals as it is not fitted yet")
