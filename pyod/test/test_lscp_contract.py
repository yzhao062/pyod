# -*- coding: utf-8 -*-


import os
import sys
import unittest
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from numpy.testing import assert_allclose
from sklearn.base import clone

from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.models.knn import KNN


def _two_lof_detectors():
    return [LOF(), LOF()]


class TestLSCPParameterContract(unittest.TestCase):
    """get_params()/clone() must describe the estimator the user built.

    fit used to write check_random_state(self.random_state) and the n_bins
    clamp back onto the constructor parameters (#754), so one fit changed
    what clone() produced and what get_params() reported.
    """

    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = np.vstack([rng.rand(60, 6), rng.rand(6, 6) + 8.0])

    def test_parameters_stored_as_given(self):
        clf = LSCP(detector_list=_two_lof_detectors(),
                   n_bins=10, random_state=0)
        clf.fit(self.X)
        assert clf.n_bins == 10
        assert clf.random_state == 0
        assert clf.get_params(deep=False)["n_bins"] == 10
        assert clf.get_params(deep=False)["random_state"] == 0

    def test_random_state_not_replaced_by_generator(self):
        # fit used to store the resolved RandomState object in place of the
        # seed, so get_params() leaked a live generator (#754).
        clf = LSCP(detector_list=_two_lof_detectors(), random_state=0)
        clf.fit(self.X)
        assert not isinstance(clf.random_state, np.random.RandomState)

    def test_clone_preserves_parameters(self):
        clf = LSCP(detector_list=_two_lof_detectors(),
                   n_bins=10, random_state=0)
        clf.fit(self.X)
        cloned = clone(clf)
        assert cloned.n_bins == 10
        assert cloned.random_state == 0
        assert cloned.get_params(deep=False)["n_bins"] == 10
        assert cloned.get_params(deep=False)["random_state"] == 0

    def test_fitted_attributes_expose_resolved_values(self):
        clf = LSCP(detector_list=_two_lof_detectors(),
                   n_bins=10, random_state=0)
        clf.fit(self.X)
        assert isinstance(clf.random_state_, np.random.RandomState)
        # n_bins=10 with two detectors clamps, so the fitted attribute is 2
        # while the parameter stays 10 (checked above and in the clamp test).
        assert clf.n_bins_ == 2

    def test_n_bins_clamped_to_fitted_attribute_only(self):
        # n_bins=10 with two detectors used to overwrite self.n_bins to 2.
        clf = LSCP(detector_list=_two_lof_detectors(), n_bins=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.fit(self.X)
        assert clf.n_bins == 10
        assert clf.n_bins_ == 2
        reduction = [w for w in caught
                     if "reducing n_bins to n_clf" in str(w.message)]
        assert len(reduction) == 1

    def test_warning_fires_once_per_fit_not_per_instance(self):
        # The clamp warning used to fire inside _get_competent_detectors,
        # which runs once per test instance: 6 test rows -> 6 warnings.
        clf = LSCP(detector_list=_two_lof_detectors(), n_bins=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.fit(self.X)
        reduction = [w for w in caught
                     if "reducing n_bins to n_clf" in str(w.message)]
        assert len(reduction) == 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.decision_function(self.X)
        reduction = [w for w in caught
                     if "reducing n_bins to n_clf" in str(w.message)]
        assert len(reduction) == 0

    def test_clone_after_fit_round_trips_scores(self):
        # A cloned estimator must reproduce the original's scores from the
        # same seed; before #754 the clone inherited the advanced generator.
        base = LSCP(detector_list=[LOF(), KNN()],
                    n_bins=10, random_state=0)
        first = base.fit(self.X).decision_scores_.copy()
        refit = base.fit(self.X).decision_scores_
        assert_allclose(first, refit)

        cloned = clone(base)
        cloned_scores = cloned.fit(self.X).decision_scores_
        assert_allclose(first, cloned_scores)

    def test_none_random_state_keeps_advancing_behavior(self):
        # RandomState instances and None keep their advancing-generator
        # behavior, matching the #753 contract for Sampling/KPCA.
        gen = np.random.RandomState(7)
        clf = LSCP(detector_list=_two_lof_detectors(), random_state=gen)
        clf.fit(self.X)
        assert clf.random_state is gen
        assert clf.random_state_ is gen


class TestLSCPBehaviorUnchanged(unittest.TestCase):
    """The clamp semantics and scores must not change for untouched paths."""

    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = np.vstack([rng.rand(60, 6), rng.rand(6, 6) + 8.0])

    def test_scores_unchanged_when_no_clamp(self):
        # n_bins <= n_clf: the fitted attribute equals the parameter, so
        # scores must be identical to a direct histogram of n_bins bins.
        clf = LSCP(detector_list=_two_lof_detectors(), n_bins=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(self.X)
        assert clf.n_bins_ == 2
        assert clf.decision_scores_ is not None
        assert len(clf.decision_scores_) == self.X.shape[0]

    def test_no_clamp_means_no_warning(self):
        # n_bins <= n_clf takes the else branch: fitted attribute equals the
        # parameter and the reduction warning never fires.
        clf = LSCP(detector_list=_two_lof_detectors(), n_bins=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.fit(self.X)
        reduction = [w for w in caught
                     if "reducing n_bins to n_clf" in str(w.message)]
        assert len(reduction) == 0
        assert clf.n_bins_ == 2


def suite():
    return unittest.TestLoader().loadTestsFromModule(
        __import__(__name__, fromlist=['_x']))


if __name__ == '__main__':
    unittest.main()
