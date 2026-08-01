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


try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


@unittest.skipUnless(_ST_AVAILABLE, "sentence-transformers not installed")
class TestAirGappedAndPreinstantiated(unittest.TestCase):

    def setUp(self):
        self.texts = [
            "normal transaction at grocery store",
            "normal payment to utility company",
            "purchase at coffee shop",
            "routine bill payment processed successfully",
            "standard payroll deposit received",
            "ANOMALOUS: wire transfer to offshore account 9x normal amount",
        ]

    def test_preinstantiated_sentencetransformer(self):
        """EmbeddingOD accepts a pre-loaded SentenceTransformer directly."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        clf = EmbeddingOD(encoder=model, detector='IForest')
        clf.fit(self.texts)
        self.assertEqual(len(clf.labels_), len(self.texts))
        scores = clf.decision_function(self.texts)
        self.assertEqual(scores.shape, (len(self.texts),))

    def test_preinstantiated_not_reloaded(self):
        """Pre-instantiated model is reused, not reloaded on each encode."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        original_id = id(model)
        clf = EmbeddingOD(encoder=model, detector='KNN')
        clf.fit(self.texts)
        from pyod.utils.encoders.sentence_transformer import (
            SentenceTransformerEncoder)
        enc = clf.encoder_
        self.assertIsInstance(enc, SentenceTransformerEncoder)
        self.assertEqual(id(enc.model_), original_id)

    def test_local_path_string(self):
        """EmbeddingOD accepts a local filesystem path string."""
        import tempfile
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            clf = EmbeddingOD(encoder=tmpdir, detector='IForest')
            clf.fit(self.texts)
            self.assertEqual(len(clf.labels_), len(self.texts))

    def test_local_path_no_network_call(self):
        """Local path loading uses local_files_only=True (no Hub call)."""
        import tempfile
        from unittest.mock import patch
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            # Patch the class reference inside the encoder module so the
            # constructor call is interceptable regardless of ST version.
            target = ('pyod.utils.encoders.sentence_transformer'
                      '.SentenceTransformer')
            with patch(target, wraps=SentenceTransformer) as mock_st:
                clf = EmbeddingOD(encoder=tmpdir, detector='KNN')
                clf.fit(self.texts)
                self.assertTrue(
                    mock_st.called,
                    "SentenceTransformer constructor was not called"
                )
                self.assertTrue(
                    mock_st.call_args.kwargs.get('local_files_only', False),
                    "SentenceTransformer not called with local_files_only=True"
                )

    def test_invalid_preinstantiated_type(self):
        """Non-string, non-BaseEncoder, non-SentenceTransformer raises."""
        with self.assertRaises(TypeError):
            from pyod.utils.encoders import resolve_encoder
            resolve_encoder(42)

    def test_resolve_encoder_sentencetransformer_instance(self):
        """resolve_encoder wraps SentenceTransformer in encoder class."""
        from pyod.utils.encoders import resolve_encoder
        from pyod.utils.encoders.sentence_transformer import (
            SentenceTransformerEncoder)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        enc = resolve_encoder(model)
        self.assertIsInstance(enc, SentenceTransformerEncoder)

    def test_resolve_st_instance_no_download(self):
        """No-download regression guard for the resolver-order bug.

        A SentenceTransformer instance must resolve to
        SentenceTransformerEncoder and never to CallableEncoder (the
        instance is callable, so resolution order matters). ``modules=[]``
        builds an empty model, so this test needs no network or model
        download, unlike the integration tests above.
        """
        from pyod.utils.encoders import CallableEncoder, resolve_encoder
        from pyod.utils.encoders.sentence_transformer import (
            SentenceTransformerEncoder)

        model = SentenceTransformer(modules=[])
        enc = resolve_encoder(model)
        self.assertIsInstance(enc, SentenceTransformerEncoder)
        self.assertNotIsInstance(enc, CallableEncoder)


class TestLocalPathEncoderFallback(unittest.TestCase):

    def test_local_path_falls_back_to_huggingface(self):
        """A local path falls back to the HuggingFace backend when
        sentence-transformers is unavailable, instead of raising."""
        import tempfile
        from unittest.mock import patch
        import pyod.utils.encoders as enc

        sentinel = object()

        def fake_create(backend, **kwargs):
            if backend == 'sentence_transformer':
                raise ImportError("sentence-transformers not installed")
            if backend == 'huggingface':
                return sentinel
            raise AssertionError("unexpected backend: %s" % backend)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(enc, '_create_encoder',
                              side_effect=fake_create):
                resolved = enc.resolve_encoder(tmpdir)

        self.assertIs(resolved, sentinel)

    def test_local_path_uses_sentence_transformer_when_available(self):
        """Local path uses the SentenceTransformer backend when available."""
        import tempfile
        from unittest.mock import patch
        import pyod.utils.encoders as enc

        sentinel = object()
        seen = []

        def fake_create(backend, **kwargs):
            seen.append(backend)
            if backend == 'sentence_transformer':
                return sentinel
            raise AssertionError("should not reach: %s" % backend)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(enc, '_create_encoder', side_effect=fake_create):
                resolved = enc.resolve_encoder(tmpdir)

        self.assertIs(resolved, sentinel)
        self.assertEqual(seen, ['sentence_transformer'])

    def test_local_path_no_backend_raises_importerror(self):
        """Local path with neither backend installed raises ImportError."""
        import tempfile
        from unittest.mock import patch
        import pyod.utils.encoders as enc

        def fake_create(backend, **kwargs):
            raise ImportError("backend not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(enc, '_create_encoder', side_effect=fake_create):
                with self.assertRaises(ImportError):
                    enc.resolve_encoder(tmpdir)


class _FakeST:
    """Stand-in for SentenceTransformer; no torch/ST install required."""

    def __init__(self, model_name=None, device=None, **kwargs):
        self.kwargs = kwargs

    def encode(self, X, **kw):
        import numpy as np
        return np.zeros((len(X), 4), dtype=float)


class TestSentenceTransformerEncoderLoading(unittest.TestCase):

    def test_local_path_loads_with_local_files_only(self):
        import tempfile
        from unittest.mock import patch
        import pyod.utils.encoders.sentence_transformer as st_mod
        from pyod.utils.encoders.sentence_transformer import (
            SentenceTransformerEncoder)
        with patch.object(st_mod, 'SentenceTransformer', _FakeST):
            with tempfile.TemporaryDirectory() as d:
                enc = SentenceTransformerEncoder(d)
                enc.encode(["a", "b"])
                self.assertTrue(enc.model_.kwargs.get('local_files_only'))

    def test_remote_name_loads_without_local_flag(self):
        from unittest.mock import patch
        import pyod.utils.encoders.sentence_transformer as st_mod
        from pyod.utils.encoders.sentence_transformer import (
            SentenceTransformerEncoder)
        with patch.object(st_mod, 'SentenceTransformer', _FakeST):
            enc = SentenceTransformerEncoder("some-remote-model")
            enc.encode(["a"])
            self.assertNotIn('local_files_only', enc.model_.kwargs)

    def test_preinstantiated_object_is_reused(self):
        from unittest.mock import patch
        import pyod.utils.encoders.sentence_transformer as st_mod
        from pyod.utils.encoders.sentence_transformer import (
            SentenceTransformerEncoder)
        with patch.object(st_mod, 'SentenceTransformer', _FakeST):
            model = _FakeST()
            enc = SentenceTransformerEncoder(model)
            enc.encode(["a"])
            self.assertIs(enc.model_, model)

    def test_invalid_model_name_type_raises(self):
        from unittest.mock import patch
        import pyod.utils.encoders.sentence_transformer as st_mod
        from pyod.utils.encoders.sentence_transformer import (
            SentenceTransformerEncoder)
        with patch.object(st_mod, 'SentenceTransformer', _FakeST):
            enc = SentenceTransformerEncoder(42)
            with self.assertRaises(TypeError):
                enc.encode(["a"])


if __name__ == '__main__':
    unittest.main()
