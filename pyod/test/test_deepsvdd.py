# -*- coding: utf-8 -*-


import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.deep_svdd import DeepSVDD
from pyod.utils.data import generate_data


class TestDeepSVDD(unittest.TestCase):
    def setUp(self):
        self.n_train = 6000
        self.n_test = 1000
        self.n_features = 300
        self.contamination = 0.1
        self.roc_floor = 0.5
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = DeepSVDD(n_features=self.n_features, epochs=10,
                            hidden_neurons=[64, 32],
                            contamination=self.contamination,
                            random_state=2021)
        self.clf_ae = DeepSVDD(n_features=self.n_features, epochs=5,
                               use_ae=True, output_activation='relu',
                               hidden_neurons=[16, 8, 4],
                               contamination=self.contamination,
                               preprocessing=False)
        self.clf.fit(self.X_train)
        self.clf_ae.fit(self.X_train)

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)
        assert (hasattr(self.clf, '_mu') and
                self.clf._mu is not None)
        assert (hasattr(self.clf, '_sigma') and
                self.clf._sigma is not None)
        assert (hasattr(self.clf, 'model_') and
                self.clf.model_ is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_scores_are_not_constant(self):
        # Hypersphere-collapse guard. Deep SVDD has a documented trivial
        # solution (Ruff et al., ICML 2018, Proposition 1): when the center
        # equals the all-zero-network output, all-zero weights are optimal
        # and every sample maps to the same point. The detector then returns
        # one constant score and is silently useless, while a ROC floor can
        # still be met by floating-point noise in the ordering. Assert score
        # diversity directly, for both the standard and autoencoder paths.
        for name, clf in (('standard', self.clf), ('autoencoder', self.clf_ae)):
            scores = np.asarray(clf.decision_scores_, dtype=float)
            n_unique = len(np.unique(np.round(scores, 8)))
            assert n_unique > 1, (
                f'{name} DeepSVDD produced {n_unique} unique score(s) for '
                f'{len(scores)} samples: the hypersphere collapsed.')

    def test_center_is_valid(self):
        # The fitted center lives in ``c_``. It must be the initialized,
        # detached vector -- not the scalar 0.0 that makes the trivial
        # solution optimal, and not a graph-carrying tensor (which breaks
        # the second backward pass).
        c = self.clf.c_
        assert not isinstance(c, float), \
            'center must not be the scalar 0.0 (trivial-solution condition)'
        assert torch.is_tensor(c), f'center should be a tensor, got {type(c)}'
        assert c.numel() > 1, 'center should be a vector in the output space'
        assert not c.requires_grad, 'center must be detached from the graph'
        assert torch.isfinite(c).all(), 'center must be finite'
        assert torch.any(c != 0), 'center must not be all zeros'

    def test_center_is_not_stored_in_constructor_param(self):
        # sklearn contract: fit() must not overwrite a constructor parameter.
        # If it does, get_params() leaks fitted state, clone() starts
        # pre-seeded instead of fresh, and a refit trains a newly built
        # network against the previous fit's center.
        assert self.clf.c is None, \
            'fit() must leave the constructor parameter `c` untouched'
        assert self.clf.get_params()['c'] is None, \
            'get_params() must not expose fitted state'
        assert clone(self.clf).c is None, 'clone() must start unfitted'

    def test_refit_reinitializes_center(self):
        # A refit builds a new network, so it must recompute the center for
        # that network. Assert the contract (the fitted center equals the new
        # inner model's own initialization) rather than "the two centers
        # differ" -- the latter would false-fail if DeepSVDD later reseeded
        # an integer random_state on each fit, which is a normal sklearn
        # determinism pattern.
        clf = DeepSVDD(n_features=self.n_features, epochs=2,
                       hidden_neurons=[64, 32],
                       contamination=self.contamination, random_state=2021)
        clf.fit(self.X_train)
        first_model = clf.model_
        clf.fit(self.X_train)
        assert clf.model_ is not first_model, 'refit did not build a new model'
        assert torch.allclose(clf.c_, clf.model_.c), \
            'c_ does not match the center the refit network initialized'

    def test_all_zero_center_is_rejected(self):
        # c=0 is exactly the trivial-solution condition of Ruff et al. 2018
        # Proposition 1 for a bias-free network; refuse it rather than
        # silently training toward a collapsed hypersphere. Cover both the
        # documented scalar form and an all-zero vector.
        for bad_c in (0.0, np.zeros(32, dtype=np.float32)):
            clf = DeepSVDD(n_features=self.n_features, epochs=1,
                           hidden_neurons=[64, 32],
                           contamination=self.contamination, c=bad_c)
            with assert_raises(ValueError):
                clf.fit(self.X_train)

    def test_custom_center_is_validated_and_copied(self):
        # A user-supplied center must be shape-checked, finite-checked, and
        # copied so that mutating the original after fit cannot silently
        # change the fitted estimator.
        width = 32

        # wrong width -> clear error instead of an opaque broadcast failure
        clf = DeepSVDD(n_features=self.n_features, epochs=1,
                       hidden_neurons=[64, 32],
                       contamination=self.contamination,
                       c=np.ones(width + 1, dtype=np.float32))
        with assert_raises(ValueError):
            clf.fit(self.X_train)

        # non-finite -> rejected (previously produced non-finite scores)
        clf = DeepSVDD(n_features=self.n_features, epochs=1,
                       hidden_neurons=[64, 32],
                       contamination=self.contamination, c=float('nan'))
        with assert_raises(ValueError):
            clf.fit(self.X_train)

        # partial zeros are legitimate and must be accepted
        partial = np.zeros(width, dtype=np.float32)
        partial[0] = 1.0
        clf = DeepSVDD(n_features=self.n_features, epochs=1,
                       hidden_neurons=[64, 32],
                       contamination=self.contamination, c=partial.copy())
        clf.fit(self.X_train)
        assert clf.c_.numel() == width

        # the fitted center must not alias the caller's array
        arr = np.ones(width, dtype=np.float32)
        clf = DeepSVDD(n_features=self.n_features, epochs=1,
                       hidden_neurons=[64, 32],
                       contamination=self.contamination, c=arr)
        clf.fit(self.X_train)
        before = clf.c_.clone()
        arr[:] = 99.0
        assert torch.equal(before, clf.c_), \
            'fitted center aliases the caller-supplied array'

    def test_fit_changes_parameters(self):
        # Guards only the original defect (issue #606): `loss.backward()`
        # commented out, so no gradient ever reached the weights. Note this
        # assertion does NOT detect hypersphere collapse -- a collapsing run
        # moves its weights too, just toward the trivial solution. Score
        # diversity (test_scores_are_not_constant) is what catches that.
        clf = DeepSVDD(n_features=self.n_features, epochs=5,
                       hidden_neurons=[64, 32],
                       contamination=self.contamination, random_state=2021)
        clf.model_ = None
        clf.fit(self.X_train)
        trained = [p.detach().clone() for p in clf.model_.parameters()]

        fresh = DeepSVDD(n_features=self.n_features, epochs=0,
                         hidden_neurons=[64, 32],
                         contamination=self.contamination, random_state=2021)
        fresh.fit(self.X_train)
        initial = [p.detach().clone() for p in fresh.model_.parameters()]

        moved = any(not torch.allclose(a, b)
                    for a, b in zip(trained, initial))
        assert moved, 'fit() did not change any model parameter'

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(self.X_test,
                                                        method='linear',
                                                        return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_with_rejection(self):
        pred_labels = self.clf.predict_with_rejection(self.X_test,
                                                      return_stats=False)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_with_rejection_stats(self):
        _, [expected_rejrate, ub_rejrate,
            ub_cost] = self.clf.predict_with_rejection(self.X_test,
                                                       return_stats=True)
        assert (expected_rejrate >= 0)
        assert (expected_rejrate <= 1)
        assert (ub_rejrate >= 0)
        assert (ub_rejrate <= 1)
        assert (ub_cost >= 0)

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        clone_clf = clone(self.clf_ae)

    def test_verbose_controls_epoch_logging(self):
        # verbose was documented and stored but never consulted, so the
        # per-epoch loss was printed even with verbose=0.
        X = np.random.RandomState(42).randn(50, 10)

        def epoch_lines(verbose):
            buffer = StringIO()
            with redirect_stdout(buffer):
                DeepSVDD(n_features=10, epochs=3, hidden_neurons=[8, 4],
                         verbose=verbose, random_state=2021).fit(X)
            return [line for line in buffer.getvalue().splitlines()
                    if line.startswith('Epoch')]

        assert_equal(len(epoch_lines(0)), 0)
        assert_equal(len(epoch_lines(1)), 3)
        # a higher verbosity must not be quieter than a lower one
        assert_equal(len(epoch_lines(2)), 3)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
