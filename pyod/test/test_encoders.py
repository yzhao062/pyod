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


from pyod.utils.encoders import resolve_encoder, _ENCODER_REGISTRY


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
