"""Shared quality assertions for time-series detector tests."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_consistent_length


def assert_scores_separate_labels(y, scores):
    """Assert that scores separate positive and negative fixture labels."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    check_consistent_length(y, scores)
    assert np.unique(scores).size > 1, 'detector returned constant scores'
    auc = roc_auc_score(y, scores)
    assert auc > 0.5, 'detector scores do not separate the fixture labels'
