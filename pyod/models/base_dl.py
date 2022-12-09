# -*- coding: utf-8 -*-
"""Base class for deep learning models
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import tensorflow


def _get_tensorflow_version():  # pragma: no cover
    """ Utility function to decide the version of tensorflow, which will 
    affect how to import keras models. 

    Returns
    -------
    tensorflow version : int

    """

    tf_version = str(tensorflow.__version__)
    if int(tf_version.split(".")[0]) != 1 and int(
            tf_version.split(".")[0]) != 2:
        raise ValueError("tensorflow version error")

    return int(tf_version.split(".")[0]) * 100 + int(tf_version.split(".")[1])
