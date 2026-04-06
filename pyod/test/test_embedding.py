# -*- coding: utf-8 -*-

import os
import sys
import unittest

import numpy as np
from numpy.testing import assert_equal
from sklearn.base import clone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.embedding import EmbeddingOD


def _mock_encoder(X):
    """Deterministic mock encoder for testing."""
    rng = np.random.RandomState(42)
    return rng.randn(len(X), 20)


class TestEmbeddingOD(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.X_train = [f"train_{i}" for i in range(self.n_train)]
        self.X_test = [f"test_{i}" for i in range(self.n_test)]

        self.clf = EmbeddingOD(encoder=_mock_encoder, detector='KNN',
                               contamination=self.contamination)
        self.clf.fit(self.X_train)

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

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.n_train)

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)
        assert_equal(pred_scores.shape[0], self.n_test)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape[0], self.n_test)
        assert set(pred_labels).issubset({0, 1})

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
        with self.assertRaises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                    return_confidence=True)
        assert_equal(pred_labels.shape[0], self.n_test)
        assert_equal(confidence.shape[0], self.n_test)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_with_rejection(self):
        pred_labels = self.clf.predict_with_rejection(self.X_test,
                                                       return_stats=False)
        assert_equal(pred_labels.shape[0], self.n_test)

    def test_detector_string_resolution(self):
        for name in ['KNN', 'LOF', 'ECOD', 'IForest', 'HBOS',
                      'COPOD', 'PCA', 'OCSVM', 'INNE']:
            clf = EmbeddingOD(encoder=_mock_encoder, detector=name)
            clf.fit(self.X_train)
            assert hasattr(clf, 'decision_scores_')

    def test_detector_instance(self):
        from pyod.models.knn import KNN
        clf = EmbeddingOD(encoder=_mock_encoder,
                          detector=KNN(n_neighbors=3))
        clf.fit(self.X_train)
        assert hasattr(clf, 'decision_scores_')

    def test_detector_instance_is_cloned(self):
        from pyod.models.knn import KNN
        original = KNN(n_neighbors=3)
        clf = EmbeddingOD(encoder=_mock_encoder, detector=original)
        clf.fit(self.X_train)
        # original should not be fitted (it was cloned)
        assert not hasattr(original, 'decision_scores_')

    def test_invalid_detector_raises(self):
        with self.assertRaises(ValueError):
            EmbeddingOD(encoder=_mock_encoder,
                        detector='NoSuchDetector').fit(self.X_train)

    def test_standardize(self):
        clf = EmbeddingOD(encoder=_mock_encoder, detector='KNN',
                          standardize=True)
        clf.fit(self.X_train)
        assert hasattr(clf, 'scaler_')

    def test_no_standardize(self):
        clf = EmbeddingOD(encoder=_mock_encoder, detector='KNN',
                          standardize=False)
        clf.fit(self.X_train)
        assert not hasattr(clf, 'scaler_')

    def test_reduce_dim(self):
        clf = EmbeddingOD(encoder=_mock_encoder, detector='KNN',
                          reduce_dim=5)
        clf.fit(self.X_train)
        assert hasattr(clf, 'pca_')
        scores = clf.decision_function(self.X_test)
        assert_equal(scores.shape[0], self.n_test)

    def test_cache_embeddings(self):
        clf = EmbeddingOD(encoder=_mock_encoder, detector='KNN',
                          cache_embeddings=True)
        clf.fit(self.X_train)
        assert hasattr(clf, 'train_embeddings_')
        assert_equal(clf.train_embeddings_.shape[0], self.n_train)

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def test_default_detector_is_lunar(self):
        clf = EmbeddingOD(encoder=_mock_encoder)
        assert clf.detector == 'LUNAR'


class TestEmbeddingODPresets(unittest.TestCase):
    def test_for_text_returns_instance(self):
        clf = EmbeddingOD.for_text(quality='fast')
        assert isinstance(clf, EmbeddingOD)
        assert clf.encoder == 'all-MiniLM-L6-v2'
        assert clf.detector == 'KNN'

    def test_for_text_balanced(self):
        clf = EmbeddingOD.for_text(quality='balanced')
        assert clf.encoder == 'all-mpnet-base-v2'
        assert clf.detector == 'LUNAR'

    def test_for_text_best(self):
        clf = EmbeddingOD.for_text(quality='best')
        assert clf.encoder == 'text-embedding-3-large'
        assert clf.detector == 'LUNAR'
        assert clf.cache_embeddings is True

    def test_for_text_override(self):
        clf = EmbeddingOD.for_text(quality='fast', detector='LOF')
        assert clf.detector == 'LOF'

    def test_for_text_invalid_quality(self):
        with self.assertRaises(ValueError):
            EmbeddingOD.for_text(quality='invalid')

    def test_for_image_returns_instance(self):
        clf = EmbeddingOD.for_image(quality='fast')
        assert isinstance(clf, EmbeddingOD)
        assert clf.encoder == 'dinov2-small'
        assert clf.detector == 'KNN'

    def test_for_image_balanced(self):
        clf = EmbeddingOD.for_image(quality='balanced')
        assert clf.encoder == 'dinov2-base'
        assert clf.detector == 'LOF'

    def test_for_image_best(self):
        clf = EmbeddingOD.for_image(quality='best')
        assert clf.encoder == 'dinov2-large'
        assert clf.detector == 'KNN'

    def test_for_image_override(self):
        clf = EmbeddingOD.for_image(quality='fast', detector='ECOD')
        assert clf.detector == 'ECOD'


import importlib


@unittest.skipUnless(
    importlib.util.find_spec('sentence_transformers') is not None,
    "sentence-transformers not installed")
class TestEmbeddingODIntegration(unittest.TestCase):
    """End-to-end test with real sentence-transformers encoder."""

    def setUp(self):
        self.normal_train = [
            "Sunny weather expected throughout the week",
            "Light rain showers predicted for tomorrow morning",
            "Temperature will reach 75 degrees today",
            "Clear skies and mild winds this afternoon",
            "A cold front will bring cooler temperatures",
            "Morning fog expected to clear by noon",
            "High pressure system bringing warm weather",
            "Partly cloudy with a chance of evening showers",
        ] * 10  # 80 normal training samples

        self.test_normal = [
            "Thunderstorms likely later this evening",
            "Weekend forecast shows pleasant conditions",
        ] * 5  # 10 normal
        self.test_anomaly = [
            "The stock market crashed by 500 points today",
            "Scientists discovered alien life on Mars",
            "The football team won the championship game",
        ]  # 3 anomalous (different topic)

        self.X_test = self.test_normal + self.test_anomaly
        self.y_test = np.array([0] * 10 + [1] * 3)

    def test_text_detection_knn(self):
        clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN',
                          contamination=0.1)
        clf.fit(self.normal_train)

        scores = clf.decision_function(self.X_test)
        assert_equal(scores.shape[0], len(self.X_test))

        labels = clf.predict(self.X_test)
        assert set(labels).issubset({0, 1})

        proba = clf.predict_proba(self.X_test)
        assert proba.min() >= 0
        assert proba.max() <= 1

    def test_for_text_preset(self):
        clf = EmbeddingOD.for_text(quality='fast')
        clf.fit(self.normal_train)
        scores = clf.decision_function(self.X_test)
        assert_equal(scores.shape[0], len(self.X_test))


from pyod.models.embedding import MultiModalOD
from pyod.models.knn import KNN


def _mock_encoder_a(X):
    rng = np.random.RandomState(10)
    return rng.randn(len(X), 15)


def _mock_encoder_b(X):
    rng = np.random.RandomState(20)
    return rng.randn(len(X), 12)


class TestMultiModalOD(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.train_data = {
            'text': [f"train_{i}" for i in range(self.n_train)],
            'tabular': np.random.RandomState(42).randn(self.n_train, 5),
        }
        self.test_data = {
            'text': [f"test_{i}" for i in range(self.n_test)],
            'tabular': np.random.RandomState(43).randn(self.n_test, 5),
        }

    def test_fit_and_predict(self):
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'tabular': KNN(),
        })
        clf.fit(self.train_data)
        assert hasattr(clf, 'decision_scores_')
        assert_equal(len(clf.decision_scores_), self.n_train)

        scores = clf.decision_function(self.test_data)
        assert_equal(scores.shape[0], self.n_test)

    def test_predict_labels(self):
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'tabular': KNN(),
        })
        clf.fit(self.train_data)
        labels = clf.predict(self.test_data)
        assert_equal(labels.shape[0], self.n_test)
        assert set(labels).issubset({0, 1})

    def test_combination_average(self):
        clf = MultiModalOD(
            modalities={
                'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
                'tabular': KNN(),
            },
            combination='average')
        clf.fit(self.train_data)
        assert hasattr(clf, 'decision_scores_')

    def test_combination_maximization(self):
        clf = MultiModalOD(
            modalities={
                'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
                'tabular': KNN(),
            },
            combination='maximization')
        clf.fit(self.train_data)
        assert hasattr(clf, 'decision_scores_')

    def test_combination_median(self):
        clf = MultiModalOD(
            modalities={
                'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
                'tabular': KNN(),
            },
            combination='median')
        clf.fit(self.train_data)
        assert hasattr(clf, 'decision_scores_')

    def test_invalid_combination_raises(self):
        clf = MultiModalOD(
            modalities={
                'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
                'tabular': KNN(),
            },
            combination='invalid')
        with self.assertRaises(ValueError):
            clf.fit(self.train_data)

    def test_missing_modality_raises(self):
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'tabular': KNN(),
        })
        with self.assertRaises(KeyError):
            clf.fit({'text': self.train_data['text']})

    def test_non_dict_input_raises(self):
        clf = MultiModalOD(modalities={
            'tabular': KNN(),
        })
        with self.assertRaises(TypeError):
            clf.fit(np.random.randn(50, 5))

    def test_three_modalities(self):
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'image': EmbeddingOD(encoder=_mock_encoder_b, detector='LOF'),
            'tabular': KNN(),
        })
        train = {
            'text': self.train_data['text'],
            'image': [f"img_{i}" for i in range(self.n_train)],
            'tabular': self.train_data['tabular'],
        }
        clf.fit(train)
        assert len(clf.detectors_) == 3

    def test_no_standardize(self):
        clf = MultiModalOD(
            modalities={
                'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
                'tabular': KNN(),
            },
            standardize_scores=False)
        clf.fit(self.train_data)
        assert hasattr(clf, 'decision_scores_')

    def test_missing_modality_at_test_time(self):
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'tabular': KNN(),
        })
        clf.fit(self.train_data)
        # At test time, text modality is missing
        scores = clf.decision_function({
            'text': None,
            'tabular': self.test_data['tabular'],
        })
        assert_equal(scores.shape[0], self.n_test)

    def test_missing_modality_score_stability(self):
        """Same sample should get same score regardless of batch size."""
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'tabular': KNN(),
        })
        clf.fit(self.train_data)

        # Score one sample with missing text
        single = {'text': None,
                  'tabular': self.test_data['tabular'][:1]}
        score_single = clf.decision_function(single)[0]

        # Score same sample in a batch of 10
        batch = {'text': None,
                 'tabular': self.test_data['tabular'][:10]}
        score_batch = clf.decision_function(batch)[0]

        # Scores should be identical (using training scalers)
        np.testing.assert_allclose(score_single, score_batch)

    def test_missing_modality_predict(self):
        """predict() should work with missing modalities."""
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'tabular': KNN(),
        })
        clf.fit(self.train_data)
        labels = clf.predict({
            'text': None,
            'tabular': self.test_data['tabular'],
        })
        assert_equal(labels.shape[0], self.n_test)
        assert set(labels).issubset({0, 1})

    def test_all_modalities_missing_raises(self):
        clf = MultiModalOD(modalities={
            'text': EmbeddingOD(encoder=_mock_encoder_a, detector='KNN'),
            'tabular': KNN(),
        })
        clf.fit(self.train_data)
        with self.assertRaises(ValueError):
            clf.decision_function({'text': None, 'tabular': None})

    def test_detectors_are_cloned(self):
        original_det = KNN()
        clf = MultiModalOD(modalities={'tabular': original_det})
        clf.fit({'tabular': self.train_data['tabular']})
        assert not hasattr(original_det, 'decision_scores_')


if __name__ == '__main__':
    unittest.main()
