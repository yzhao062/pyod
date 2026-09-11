"""Shared quality assertions for time-series detector tests."""

import numpy as np
from sklearn.utils import check_consistent_length

from pyod.utils.utility import detection_lift


def assert_scores_separate_labels(y, scores):
    """Assert that scores separate positive and negative fixture labels."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    check_consistent_length(y, scores)
    assert np.unique(np.round(scores, 8)).size > 1, \
        'detector returned constant scores'
    lift = detection_lift(y, scores)
    assert lift > 1.1, 'detector scores do not separate the fixture labels'
