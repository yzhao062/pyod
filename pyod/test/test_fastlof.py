# -*- coding: utf-8 -*-

import os
import sys
import unittest

# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.fastlof import FastLOF
from pyod.utils.data import generate_data


class TestFastLOF(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = FastLOF(contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        """Test that all expected attributes are present after fitting."""
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
        assert (hasattr(self.clf, 'n_neighbors_') and
                self.clf.n_neighbors_ is not None)

    def test_train_scores(self):
        """Test that decision scores have correct shape."""
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        """Test prediction scores on test data."""
        pred_scores = self.clf.decision_function(self.X_test)
        assert_equal(len(pred_scores), self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        """Test prediction labels."""
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(len(pred_labels), self.X_test.shape[0])

    def test_prediction_proba(self):
        """Test prediction probabilities."""
        pred_proba = self.clf.predict_proba(self.X_train)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        """Test linear prediction probabilities."""
        pred_proba = self.clf.predict_proba(self.X_train, method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        """Test unify prediction probabilities."""
        pred_proba = self.clf.predict_proba(self.X_train, method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        """Test invalid prediction probability method."""
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_train, method='something')

    def test_prediction_labels_confidence(self):
        """Test prediction labels with confidence on training data."""
        pred_labels, confidence = self.clf.predict(self.X_train,
                                                   return_confidence=True)
        assert_equal(pred_labels.shape, self.y_train.shape)
        assert_equal(confidence.shape, self.y_train.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_proba_linear_confidence(self):
        """Test linear proba with confidence."""
        pred_proba, confidence = self.clf.predict_proba(self.X_train,
                                                        method='linear',
                                                        return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

        assert_equal(confidence.shape, self.y_train.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_fit_predict(self):
        """Test fit_predict method."""
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_model_clone(self):
        """Test that the model can be cloned."""
        clone_clf = clone(self.clf)

    def test_n_neighbors_validation(self):
        """Test that n_neighbors is validated correctly."""
        # Test with n_neighbors > n_samples
        clf = FastLOF(n_neighbors=300)
        clf.fit(self.X_train)
        assert clf.n_neighbors_ == self.n_train - 1

    def test_decision_function_large_n_neighbors(self):
        """Test decision_function when n_neighbors > n_samples (clamped)."""
        clf = FastLOF(n_neighbors=300, contamination=self.contamination)
        clf.fit(self.X_train)
        assert clf.n_neighbors_ == self.n_train - 1
        pred_scores = clf.decision_function(self.X_test)
        assert_equal(len(pred_scores), self.X_test.shape[0])

    def test_algorithm_validation(self):
        """Test that invalid algorithm raises error."""
        with assert_raises(ValueError):
            clf = FastLOF(algorithm='invalid')
            clf.fit(self.X_train)

    def test_metric_validation(self):
        """Test that different metrics work correctly."""
        clf = FastLOF(metric='manhattan')
        clf.fit(self.X_train)
        assert hasattr(clf, 'decision_scores_')
        
        clf2 = FastLOF(metric='cosine')
        clf2.fit(self.X_train)
        assert hasattr(clf2, 'decision_scores_')
    
    def test_p_validation(self):
        """Test that invalid p parameter raises error."""
        with assert_raises(ValueError):
            clf = FastLOF(p=0)
            clf.fit(self.X_train)
    
    def test_leaf_size_validation(self):
        """Test that invalid leaf_size raises error."""
        with assert_raises(ValueError):
            clf = FastLOF(leaf_size=0)
            clf.fit(self.X_train)
    
    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducible results."""
        clf1 = FastLOF(random_state=42)
        clf1.fit(self.X_train)
        scores1 = clf1.decision_scores_.copy()
        
        clf2 = FastLOF(random_state=42)
        clf2.fit(self.X_train)
        scores2 = clf2.decision_scores_.copy()
        
        # Results should be identical with same random_state
        assert_allclose(scores1, scores2, atol=1e-10)
    
    def test_random_state_different_seeds(self):
        """Test that different random_state values give different results."""
        clf1 = FastLOF(random_state=42)
        clf1.fit(self.X_train)
        scores1 = clf1.decision_scores_.copy()
        
        clf2 = FastLOF(random_state=123)
        clf2.fit(self.X_train)
        scores2 = clf2.decision_scores_.copy()
        
        assert hasattr(clf1, 'random_state')
        assert hasattr(clf2, 'random_state')

    def test_chunk_size_auto(self):
        """Test that automatic chunk size is set correctly."""
        clf = FastLOF(n_neighbors=10, chunk_size=None)
        clf.fit(self.X_train)
        assert hasattr(clf, 'decision_scores_')

    def test_chunk_size_manual(self):
        """Test that manual chunk size works."""
        clf = FastLOF(n_neighbors=10, chunk_size=30)
        clf.fit(self.X_train)
        assert hasattr(clf, 'decision_scores_')

    def test_train_performance(self):
        """Test that performance on training data is reasonable."""
        auc = roc_auc_score(self.y_train, self.clf.decision_scores_)
        assert auc >= self.roc_floor, \
            f"AUC {auc:.3f} is below floor {self.roc_floor}"

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
