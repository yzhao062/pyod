# -*- coding: utf-8 -*-
"""Save / load / clone round-trip coverage for PyOD detectors (issue #269).

For every core, dependency-light (torch-free) detector this module checks:

* ``save`` -> ``load(trusted=True)`` reproduces the fitted state
  (``decision_scores_``, ``labels_``, ``threshold_``) and the
  ``decision_function`` output on held-out data, following
  ``examples/save_load_model_example.py`` and honoring the ``trusted=True``
  contract added in v3.6.2 (CVE-2026-15529).
* ``sklearn.base.clone`` returns an *unfitted* estimator whose
  ``get_params`` match the original and that reproduces the original scores
  after re-fitting. This deepens the existing per-model ``test_model_clone``
  checks, which only asserted that ``clone`` did not raise.

``test_persistence.py`` already exercises the ``save`` / ``load`` /
``compat_load`` machinery in depth on IForest and ECOD; this module is the
breadth counterpart, sweeping the detector zoo.

Deep-learning (torch) and graph (PyG) detectors are intentionally out of
scope here: they carry heavier, stochastic state, need optional
dependencies, and are already skipped from the always-on torch-free lane
(see ``conftest.py``). Extending this sweep to them is a natural follow-up.
"""

import inspect
import os
import sys
import tempfile
import unittest
import warnings

from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from sklearn.base import clone

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
)

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.qmcd import QMCD
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.utils.data import generate_data
from pyod.utils.persistence import load
from pyod.utils.persistence import save

_RANDOM_STATE = 42

# Core torch-free detectors whose fitted state round-trips through both
# ``save`` / ``load`` and ``clone`` + re-fit deterministically.
CORE_DETECTORS = [
    ABOD,
    CBLOF,
    COF,
    COPOD,
    ECOD,
    GMM,
    HBOS,
    IForest,
    INNE,
    KDE,
    KNN,
    LMDD,
    LOF,
    MCD,
    OCSVM,
    PCA,
    QMCD,
    ROD,
    SOD,
    SOS,
    LODA,
]

# Detectors that ``save`` / ``load`` exactly but whose ``clone`` + re-fit is
# stochastic (Sampling draws a random subset) or whose parameters do not
# round-trip identically under ``get_params`` (KPCA). These are covered for
# persistence only; asserting clone determinism on them would be flaky.
SAVE_LOAD_ONLY_DETECTORS = [KPCA, Sampling]


def _instantiate(detector_cls):
    """Construct a detector with a fixed ``random_state`` when supported."""
    if 'random_state' in inspect.signature(detector_cls).parameters:
        return detector_cls(random_state=_RANDOM_STATE)
    return detector_cls()


class TestSaveLoadClone(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=200,
            n_test=100,
            n_features=6,
            contamination=0.1,
            random_state=_RANDOM_STATE,
        )
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _fit(self, detector_cls):
        clf = _instantiate(detector_cls)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(self.X_train)
        return clf

    def _score(self, clf):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return clf.decision_function(self.X_test)

    def _check_save_load(self, detector_cls):
        clf = self._fit(detector_cls)
        expected = self._score(clf)
        path = os.path.join(
            self._tmpdir.name, detector_cls.__name__ + '.pyod.joblib'
        )

        save(clf, path)
        # load() requires trusted=True after the v3.6.2 persistence CVE.
        loaded = load(path, trusted=True)

        self.assertIsInstance(loaded, detector_cls)
        assert_allclose(
            loaded.decision_scores_,
            clf.decision_scores_,
            rtol=1e-12,
            atol=1e-12,
        )
        assert_array_equal(loaded.labels_, clf.labels_)
        assert_allclose(
            loaded.threshold_, clf.threshold_, rtol=1e-12, atol=1e-12
        )
        assert_allclose(self._score(loaded), expected, rtol=1e-12, atol=1e-12)

    def test_save_load_roundtrip(self):
        for detector_cls in CORE_DETECTORS + SAVE_LOAD_ONLY_DETECTORS:
            with self.subTest(detector=detector_cls.__name__):
                self._check_save_load(detector_cls)

    def test_clone_returns_unfitted_equivalent(self):
        for detector_cls in CORE_DETECTORS:
            with self.subTest(detector=detector_cls.__name__):
                clf = self._fit(detector_cls)
                cloned = clone(clf)
                self.assertIsNone(
                    getattr(cloned, 'decision_scores_', None),
                    f'{detector_cls.__name__}: clone must be unfitted',
                )
                self.assertEqual(cloned.get_params(), clf.get_params())

    def test_clone_refit_reproduces_scores(self):
        for detector_cls in CORE_DETECTORS:
            with self.subTest(detector=detector_cls.__name__):
                clf = self._fit(detector_cls)
                expected = self._score(clf)
                cloned = clone(clf)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cloned.fit(self.X_train)
                assert_allclose(
                    self._score(cloned),
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f'{detector_cls.__name__}: re-fit clone diverged',
                )


if __name__ == '__main__':
    unittest.main(verbosity=2)
