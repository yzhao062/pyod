# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np
from numpy.testing import assert_equal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.encoders import BaseEncoder, CallableEncoder


class TestBaseEncoder(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseEncoder()


class TestCallableEncoder(unittest.TestCase):
    def setUp(self):
        self.n_samples = 50
        self.n_features = 10
        self.X = [f"sample_{i}" for i in range(self.n_samples)]

        def mock_fn(X):
            rng = np.random.RandomState(42)
            return rng.randn(len(X), self.n_features)

        self.encoder = CallableEncoder(fn=mock_fn)

    def test_encode_shape(self):
        emb = self.encoder.encode(self.X)
        assert_equal(emb.shape, (self.n_samples, self.n_features))

    def test_encode_dtype(self):
        emb = self.encoder.encode(self.X)
        assert emb.dtype == np.float64

    def test_encode_1d_reshaped(self):
        encoder = CallableEncoder(fn=lambda X: np.ones(len(X)))
        emb = encoder.encode(["a", "b", "c"])
        assert_equal(emb.shape, (3, 1))

    def test_encode_not_callable_raises(self):
        with self.assertRaises(TypeError):
            CallableEncoder(fn="not_callable")

    def test_encode_wrong_row_count_raises(self):
        encoder = CallableEncoder(fn=lambda X: np.ones((1, 2)))
        with self.assertRaises(ValueError):
            encoder.encode(["a", "b", "c"])


import importlib
from unittest.mock import patch, MagicMock


@unittest.skipUnless(
    importlib.util.find_spec('sentence_transformers') is not None,
    "sentence-transformers not installed")
class TestSentenceTransformerEncoder(unittest.TestCase):
    def setUp(self):
        from pyod.utils.encoders.sentence_transformer import \
            SentenceTransformerEncoder
        self.SentenceTransformerEncoder = SentenceTransformerEncoder
        self.encoder = SentenceTransformerEncoder(
            model_name='all-MiniLM-L6-v2')
        self.texts = ["The stock market rose sharply today",
                      "Heavy rain caused flooding in the city",
                      "Scientists discovered a new species of frog"]

    def test_encode_shape(self):
        emb = self.encoder.encode(self.texts)
        assert_equal(emb.shape[0], 3)
        # MiniLM produces 384-dim embeddings
        assert_equal(emb.shape[1], 384)

    def test_encode_dtype(self):
        emb = self.encoder.encode(self.texts)
        assert emb.dtype == np.float64

    def test_encode_batch_size(self):
        texts = [f"text {i}" for i in range(100)]
        emb = self.encoder.encode(texts, batch_size=16)
        assert_equal(emb.shape[0], 100)

    def test_encode_single(self):
        emb = self.encoder.encode(["hello"])
        assert_equal(emb.shape[0], 1)

    def test_normalize(self):
        enc = self.SentenceTransformerEncoder(
            model_name='all-MiniLM-L6-v2', normalize=True)
        emb = enc.encode(self.texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestOpenAIEncoder(unittest.TestCase):
    """Tests using mocked OpenAI API (no API key needed)."""

    def _make_mock_response(self, n_samples, n_dim=1536):
        """Create a mock OpenAI embeddings response."""
        response = MagicMock()
        data = []
        rng = np.random.RandomState(42)
        for i in range(n_samples):
            item = MagicMock()
            item.embedding = rng.randn(n_dim).tolist()
            data.append(item)
        response.data = data
        return response

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('pyod.utils.encoders.openai_encoder.OpenAI')
    def test_encode_shape(self, mock_openai_cls):
        from pyod.utils.encoders.openai_encoder import OpenAIEncoder

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = \
            self._make_mock_response(3)

        encoder = OpenAIEncoder(model_name='text-embedding-3-small')
        emb = encoder.encode(["text1", "text2", "text3"])
        assert_equal(emb.shape, (3, 1536))

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('pyod.utils.encoders.openai_encoder.OpenAI')
    def test_encode_batching(self, mock_openai_cls):
        from pyod.utils.encoders.openai_encoder import OpenAIEncoder

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def side_effect(**kwargs):
            n = len(kwargs['input'])
            return self._make_mock_response(n)

        mock_client.embeddings.create.side_effect = side_effect

        encoder = OpenAIEncoder(model_name='text-embedding-3-small')
        texts = [f"text_{i}" for i in range(3000)]
        emb = encoder.encode(texts)
        assert_equal(emb.shape[0], 3000)
        assert_equal(mock_client.embeddings.create.call_count, 2)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('pyod.utils.encoders.openai_encoder.OpenAI')
    def test_encode_dtype(self, mock_openai_cls):
        from pyod.utils.encoders.openai_encoder import OpenAIEncoder

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = \
            self._make_mock_response(2)

        encoder = OpenAIEncoder()
        emb = encoder.encode(["a", "b"])
        assert emb.dtype == np.float64


from pyod.utils.encoders import resolve_encoder, MultiModalEncoder, \
    _ENCODER_REGISTRY


class TestMultiModalEncoder(unittest.TestCase):
    def setUp(self):
        self.n_samples = 20
        self.text_encoder = CallableEncoder(
            fn=lambda X: np.random.RandomState(1).randn(len(X), 10))
        self.image_encoder = CallableEncoder(
            fn=lambda X: np.random.RandomState(2).randn(len(X), 8))
        self.data = {
            'text': [f"text_{i}" for i in range(self.n_samples)],
            'image': [f"img_{i}" for i in range(self.n_samples)],
            'tabular': np.random.randn(self.n_samples, 5),
        }

    def test_two_modalities(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'image': self.image_encoder,
        })
        emb = enc.encode({'text': self.data['text'],
                          'image': self.data['image']})
        assert_equal(emb.shape, (self.n_samples, 18))  # 10 + 8

    def test_passthrough(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'tabular': 'passthrough',
        })
        emb = enc.encode({'text': self.data['text'],
                          'tabular': self.data['tabular']})
        assert_equal(emb.shape, (self.n_samples, 15))  # 10 + 5

    def test_three_modalities(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'image': self.image_encoder,
            'tabular': 'passthrough',
        })
        emb = enc.encode(self.data)
        assert_equal(emb.shape, (self.n_samples, 23))  # 10 + 8 + 5

    def test_weights(self):
        enc = MultiModalEncoder(
            {'text': self.text_encoder, 'tabular': 'passthrough'},
            weights={'text': 0.5, 'tabular': 2.0})
        emb = enc.encode({'text': self.data['text'],
                          'tabular': self.data['tabular']})
        assert_equal(emb.shape, (self.n_samples, 15))

    def test_missing_modality_raises(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'image': self.image_encoder,
        })
        with self.assertRaises(KeyError):
            enc.encode({'text': self.data['text']})

    def test_non_dict_input_raises(self):
        enc = MultiModalEncoder({'text': self.text_encoder})
        with self.assertRaises(TypeError):
            enc.encode(["not", "a", "dict"])

    def test_mismatched_sample_count_raises(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'tabular': 'passthrough',
        })
        with self.assertRaises(ValueError):
            enc.encode({'text': self.data['text'],
                        'tabular': np.random.randn(5, 3)})

    def test_empty_encoders_raises(self):
        with self.assertRaises(ValueError):
            MultiModalEncoder({})

    def test_string_encoder_resolution(self):
        # String encoders are resolved lazily at encode time
        enc = MultiModalEncoder({
            'a': lambda X: np.ones((len(X), 3)),
            'b': 'passthrough',
        })
        emb = enc.encode({'a': ["x", "y"], 'b': np.array([[1], [2]])})
        assert_equal(emb.shape, (2, 4))  # 3 + 1

    def test_missing_samples_imputed(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'image': self.image_encoder,
        })
        # First call without missing to store means
        full_data = {
            'text': [f"t_{i}" for i in range(5)],
            'image': [f"img_{i}" for i in range(5)],
        }
        enc.fit_encode(full_data)

        # Second call with missing samples
        data = {
            'text': [f"t_{i}" for i in range(5)],
            'image': ["img_0", None, "img_2", None, "img_4"],
        }
        emb = enc.encode(data)
        assert_equal(emb.shape, (5, 18))  # 10 + 8
        # Missing image samples (1, 3) should be filled with training mean
        np.testing.assert_allclose(emb[1, 10:], enc.means_['image'])
        np.testing.assert_allclose(emb[3, 10:], enc.means_['image'])
        # Present samples should not equal mean
        assert not np.allclose(emb[0, 10:], enc.means_['image'])

    def test_missing_passthrough_mean_fill(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'nums': 'passthrough',
        })
        # fit_encode to store means
        full = {
            'text': [f"t_{i}" for i in range(4)],
            'nums': np.array([[1.0, 2.0], [3.0, 4.0],
                              [5.0, 6.0], [7.0, 8.0]]),
        }
        enc.fit_encode(full)
        expected_mean = np.array([4.0, 5.0])  # mean of [1,3,5,7], [2,4,6,8]

        data = {
            'text': [f"t_{i}" for i in range(4)],
            'nums': [np.array([1.0, 2.0]), None,
                     np.array([5.0, 6.0]), None],
        }
        emb = enc.encode(data)
        assert_equal(emb.shape, (4, 12))  # 10 + 2
        # Missing passthrough samples should be training mean
        np.testing.assert_allclose(emb[1, 10:], expected_mean)
        np.testing.assert_allclose(emb[3, 10:], expected_mean)
        # Present should keep values
        np.testing.assert_allclose(emb[0, 10:], [1.0, 2.0])

    def test_missing_with_weights_no_double_apply(self):
        enc = MultiModalEncoder(
            {'nums': 'passthrough'},
            weights={'nums': 3.0})
        # fit_encode to store means
        enc.fit_encode({'nums': np.array([[1.0, 2.0], [3.0, 4.0]])})
        # Mean should be [2.0, 3.0] (unweighted)
        np.testing.assert_allclose(enc.means_['nums'], [2.0, 3.0])

        # Encode with missing sample
        emb = enc.encode({'nums': [np.array([10.0, 20.0]), None]})
        # Present row: [10, 20] * 3.0 = [30, 60]
        np.testing.assert_allclose(emb[0], [30.0, 60.0])
        # Missing row: mean [2, 3] * 3.0 = [6, 9] (weight applied once)
        np.testing.assert_allclose(emb[1], [6.0, 9.0])

    def test_all_none_raises(self):
        enc = MultiModalEncoder({
            'text': self.text_encoder,
        })
        with self.assertRaises(ValueError):
            enc.encode({'text': [None, None, None]})

    def test_with_embedding_od(self):
        from pyod.models.embedding import EmbeddingOD
        enc = MultiModalEncoder({
            'text': self.text_encoder,
            'tabular': 'passthrough',
        })
        clf = EmbeddingOD(encoder=enc, detector='KNN')
        clf.fit(self.data)
        scores = clf.decision_function(self.data)
        assert_equal(scores.shape[0], self.n_samples)


class TestResolveEncoder(unittest.TestCase):
    def test_resolve_callable(self):
        fn = lambda X: np.random.randn(len(X), 5)
        encoder = resolve_encoder(fn)
        assert isinstance(encoder, CallableEncoder)

    def test_resolve_base_encoder_instance(self):
        enc = CallableEncoder(fn=lambda X: np.random.randn(len(X), 5))
        resolved = resolve_encoder(enc)
        assert resolved is enc

    def test_resolve_unknown_string(self):
        # When sentence-transformers is installed, any string auto-resolves
        # to a SentenceTransformerEncoder (lazy model load). When it is not
        # installed, raises ValueError or ImportError.
        try:
            encoder = resolve_encoder('definitely-not-a-real-model-xyz-999')
            assert hasattr(encoder, 'encode')
        except (ValueError, ImportError):
            pass

    def test_resolve_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_encoder(12345)

    def test_registry_keys_exist(self):
        expected_keys = ['all-MiniLM-L6-v2', 'text-embedding-3-small',
                         'dinov2-small', 'bert-base-uncased']
        for key in expected_keys:
            assert key in _ENCODER_REGISTRY, \
                f"'{key}' missing from registry"

    @unittest.skipUnless(
        importlib.util.find_spec('sentence_transformers') is not None,
        "sentence-transformers not installed")
    def test_resolve_registry_shortcut(self):
        encoder = resolve_encoder('all-MiniLM-L6-v2')
        assert hasattr(encoder, 'encode')

    @unittest.skipUnless(
        importlib.util.find_spec('sentence_transformers') is not None,
        "sentence-transformers not installed")
    def test_resolve_auto_sentence_transformer(self):
        encoder = resolve_encoder(
            'sentence-transformers/all-MiniLM-L6-v2')
        assert hasattr(encoder, 'encode')


if __name__ == '__main__':
    unittest.main()
