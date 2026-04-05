# EmbeddingOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `EmbeddingOD` to PyOD -- a wrapper that chains any embedding encoder (sentence-transformers, OpenAI, HuggingFace, or callable) with any existing PyOD detector, enabling text and image anomaly detection through the standard PyOD API.

**Architecture:** Thin wrapper inheriting `BaseDetector`. An encoder abstraction (`BaseEncoder`) converts raw data (text/images) to numpy embeddings; `EmbeddingOD.fit()` encodes, preprocesses (StandardScaler, optional PCA), then delegates to any PyOD detector. Follows the same delegation pattern as `pyod/models/lof.py`.

**Tech Stack:** Python 3.9+, numpy, scikit-learn (StandardScaler, PCA, check_array, check_is_fitted). Optional: sentence-transformers, openai, transformers+torch.

**Spec:** `docs/superpowers/specs/2026-04-04-embedding-od-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyod/utils/encoders/__init__.py` | BaseEncoder, CallableEncoder, resolve_encoder(), encoder registry |
| `pyod/utils/encoders/sentence_transformer.py` | SentenceTransformerEncoder |
| `pyod/utils/encoders/openai_encoder.py` | OpenAIEncoder |
| `pyod/utils/encoders/huggingface.py` | HuggingFaceEncoder |
| `pyod/models/embedding.py` | EmbeddingOD class, resolve_detector(), presets |
| `pyod/test/test_embedding.py` | EmbeddingOD tests (no external deps, uses CallableEncoder) |
| `pyod/test/test_encoders.py` | Encoder unit tests (skipUnless for optional deps) |
| `examples/embedding_od_example.py` | Usage examples |

---

### Task 1: BaseEncoder and CallableEncoder

**Files:**
- Create: `pyod/utils/encoders/__init__.py`

- [ ] **Step 1: Write the test for BaseEncoder and CallableEncoder**

Create `pyod/test/test_encoders.py`:

```python
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


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_encoders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyod.utils.encoders'`

- [ ] **Step 3: Implement BaseEncoder and CallableEncoder**

Create `pyod/utils/encoders/__init__.py`:

```python
# -*- coding: utf-8 -*-
"""Encoder abstraction for EmbeddingOD.

Provides BaseEncoder and concrete implementations for converting
raw data (text, images) to numeric embeddings.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import abc
import importlib

import numpy as np
from sklearn.utils import check_array


class BaseEncoder(abc.ABC):
    """Abstract base class for embedding encoders.

    All encoders must implement the ``encode`` method, which converts
    raw input data to a 2D numpy array of shape (n_samples, n_features).
    """

    @abc.abstractmethod
    def encode(self, X, batch_size=32, show_progress=True):
        """Convert raw input to numeric embeddings.

        Parameters
        ----------
        X : list or array-like
            Raw input data.

        batch_size : int, optional (default=32)
            Batch size for encoding.

        show_progress : bool, optional (default=True)
            Whether to show a progress bar.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, n_features)
        """

    def _validate_output(self, embeddings):
        """Validate and normalize encoder output."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        check_array(embeddings)
        return embeddings


class CallableEncoder(BaseEncoder):
    """Encoder that wraps a user-provided callable.

    Parameters
    ----------
    fn : callable
        A function that accepts raw input and returns a numpy array
        of shape (n_samples, n_features).

    Examples
    --------
    >>> import numpy as np
    >>> encoder = CallableEncoder(fn=lambda X: np.random.randn(len(X), 10))
    >>> embeddings = encoder.encode(["hello", "world"])
    >>> embeddings.shape
    (2, 10)
    """

    def __init__(self, fn):
        if not callable(fn):
            raise TypeError("fn must be callable, got %s" % type(fn))
        self.fn = fn

    def encode(self, X, batch_size=32, show_progress=True):
        embeddings = self.fn(X)
        return self._validate_output(embeddings)


# ---- Encoder registry and resolution ----

_ENCODER_REGISTRY = {
    # Sentence Transformers
    'all-MiniLM-L6-v2': ('sentence_transformer',
                         {'model_name': 'all-MiniLM-L6-v2'}),
    'all-mpnet-base-v2': ('sentence_transformer',
                          {'model_name': 'all-mpnet-base-v2'}),
    # OpenAI
    'text-embedding-3-small': ('openai',
                               {'model_name': 'text-embedding-3-small'}),
    'text-embedding-3-large': ('openai',
                               {'model_name': 'text-embedding-3-large'}),
    # HuggingFace Vision
    'dinov2-small': ('huggingface',
                     {'model_name': 'facebook/dinov2-small',
                      'modality': 'image'}),
    'dinov2-base': ('huggingface',
                    {'model_name': 'facebook/dinov2-base',
                     'modality': 'image'}),
    'dinov2-large': ('huggingface',
                     {'model_name': 'facebook/dinov2-large',
                      'modality': 'image'}),
    'clip-vit-base': ('huggingface',
                      {'model_name': 'openai/clip-vit-base-patch32',
                       'modality': 'image'}),
    # HuggingFace Text
    'bert-base-uncased': ('huggingface',
                          {'model_name': 'bert-base-uncased',
                           'modality': 'text'}),
}

_ENCODER_BACKENDS = {
    'sentence_transformer': (
        'pyod.utils.encoders.sentence_transformer',
        'SentenceTransformerEncoder'),
    'openai': (
        'pyod.utils.encoders.openai_encoder',
        'OpenAIEncoder'),
    'huggingface': (
        'pyod.utils.encoders.huggingface',
        'HuggingFaceEncoder'),
}


def _create_encoder(backend, **kwargs):
    """Create an encoder from a backend name and kwargs."""
    module_path, class_name = _ENCODER_BACKENDS[backend]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def resolve_encoder(encoder):
    """Resolve an encoder from various input types.

    Parameters
    ----------
    encoder : str, BaseEncoder, or callable
        - If BaseEncoder instance, returned as-is.
        - If callable, wrapped in CallableEncoder.
        - If string, looked up in the encoder registry. If not found,
          tries sentence-transformers first, then HuggingFace AutoModel.

    Returns
    -------
    encoder : BaseEncoder
    """
    if isinstance(encoder, BaseEncoder):
        return encoder

    if callable(encoder) and not isinstance(encoder, str):
        return CallableEncoder(encoder)

    if isinstance(encoder, str):
        # Check registry
        if encoder in _ENCODER_REGISTRY:
            backend, kwargs = _ENCODER_REGISTRY[encoder]
            return _create_encoder(backend, **kwargs)

        # Auto-resolve: try sentence-transformers first (most text
        # embedding models are compatible), then HuggingFace AutoModel
        try:
            return _create_encoder('sentence_transformer',
                                   model_name=encoder)
        except (ImportError, Exception):
            pass

        try:
            return _create_encoder('huggingface',
                                   model_name=encoder,
                                   modality='auto')
        except (ImportError, Exception):
            pass

        raise ValueError(
            "Cannot resolve encoder '%s'. Provide a registry shortcut "
            "(e.g., 'all-MiniLM-L6-v2'), a HuggingFace model ID, a "
            "BaseEncoder instance, or a callable." % encoder)

    raise TypeError("encoder must be str, BaseEncoder, or callable, "
                    "got %s" % type(encoder))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_encoders.py::TestBaseEncoder -v && python -m pytest pyod/test/test_encoders.py::TestCallableEncoder -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```
git add pyod/utils/encoders/__init__.py pyod/test/test_encoders.py
git commit -m "feat: add BaseEncoder, CallableEncoder, and encoder registry"
```

---

### Task 2: EmbeddingOD Wrapper and resolve_detector

**Files:**
- Create: `pyod/models/embedding.py`

- [ ] **Step 1: Write the test for EmbeddingOD**

Create `pyod/test/test_embedding.py`:

```python
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


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyod.models.embedding'`

- [ ] **Step 3: Implement EmbeddingOD and resolve_detector**

Create `pyod/models/embedding.py`:

```python
# -*- coding: utf-8 -*-
"""EmbeddingOD: Anomaly detection via foundation model embeddings.

Chains any embedding encoder with any PyOD detector, enabling
anomaly detection on text, image, and other non-tabular data
through PyOD's standard API.

This implements the two-step approach shown to outperform end-to-end
methods in NLP-ADBench (Li et al., EMNLP 2025) and TAD-Bench
(Cao et al., 2025).
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import importlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ..utils.encoders import resolve_encoder

_DETECTOR_SHORTCUTS = {
    'KNN': ('pyod.models.knn', 'KNN', {}),
    'LOF': ('pyod.models.lof', 'LOF', {}),
    'ECOD': ('pyod.models.ecod', 'ECOD', {}),
    'COPOD': ('pyod.models.copod', 'COPOD', {}),
    'HBOS': ('pyod.models.hbos', 'HBOS', {}),
    'PCA': ('pyod.models.pca', 'PCA', {}),
    'OCSVM': ('pyod.models.ocsvm', 'OCSVM', {}),
    'MCD': ('pyod.models.mcd', 'MCD', {}),
    'IForest': ('pyod.models.iforest', 'IForest', {}),
    'INNE': ('pyod.models.inne', 'INNE', {}),
    'ABOD': ('pyod.models.abod', 'ABOD', {}),
    'CBLOF': ('pyod.models.cblof', 'CBLOF', {}),
    'COF': ('pyod.models.cof', 'COF', {}),
    'SOD': ('pyod.models.sod', 'SOD', {}),
    'LODA': ('pyod.models.loda', 'LODA', {}),
    'AutoEncoder': ('pyod.models.auto_encoder', 'AutoEncoder', {}),
    'VAE': ('pyod.models.vae', 'VAE', {}),
    'DeepSVDD': ('pyod.models.deep_svdd', 'DeepSVDD', {}),
    'LUNAR': ('pyod.models.lunar', 'LUNAR', {}),
    'DIF': ('pyod.models.dif', 'DIF', {}),
    'GMM': ('pyod.models.gmm', 'GMM', {}),
    'KDE': ('pyod.models.kde', 'KDE', {}),
    'LMDD': ('pyod.models.lmdd', 'LMDD', {}),
    'LOCI': ('pyod.models.loci', 'LOCI', {}),
}


def resolve_detector(detector, contamination=0.1):
    """Resolve a detector from a string name or BaseDetector instance.

    Parameters
    ----------
    detector : str or BaseDetector
        If string, creates a default-configured instance from the
        shortcut registry. If BaseDetector, returned as-is.

    contamination : float, optional (default=0.1)
        Contamination parameter passed to newly created detectors.

    Returns
    -------
    detector : BaseDetector
    """
    if isinstance(detector, BaseDetector):
        return detector

    if isinstance(detector, str):
        if detector not in _DETECTOR_SHORTCUTS:
            raise ValueError(
                "Unknown detector '%s'. Available: %s"
                % (detector, list(_DETECTOR_SHORTCUTS.keys())))
        module_path, class_name, kwargs = _DETECTOR_SHORTCUTS[detector]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(contamination=contamination, **kwargs)

    raise TypeError("detector must be str or BaseDetector, got %s"
                    % type(detector))


class EmbeddingOD(BaseDetector):
    """Anomaly detection on raw data via embedding + detector pipeline.

    Chains any embedding encoder with any PyOD detector. Encode raw data
    (text, images, or other modalities) into numeric embeddings, then
    apply outlier detection in the embedding space.

    This implements the two-step approach shown to outperform end-to-end
    methods in NLP-ADBench (Li et al., EMNLP 2025) and TAD-Bench
    (Cao et al., 2025).

    Parameters
    ----------
    encoder : str, BaseEncoder, or callable
        Embedding encoder. Accepts:
        - Registry shortcut: 'all-MiniLM-L6-v2', 'text-embedding-3-small',
          'dinov2-base'
        - HuggingFace model ID: 'sentence-transformers/all-MiniLM-L6-v2'
        - BaseEncoder instance
        - Callable: fn(X) -> np.ndarray of shape (n_samples, n_features)

    detector : str or BaseDetector, optional (default='LUNAR')
        Any PyOD detector. String resolves to default-configured instance.
        Default is LUNAR (best performer in NLP-ADBench).

    contamination : float, optional (default=0.1)
        Expected proportion of outliers in the dataset. Must be in (0, 0.5].

    batch_size : int, optional (default=32)
        Batch size for encoding.

    cache_embeddings : bool, optional (default=False)
        Cache training embeddings to avoid re-encoding.
        Recommended for API-based encoders (e.g., OpenAI).

    reduce_dim : int or None, optional (default=None)
        If set, apply PCA to reduce embedding dimensionality before
        detection. Recommended for embeddings >1000 dims with
        distance-based detectors (KNN, LOF).

    standardize : bool, optional (default=True)
        Apply StandardScaler to embeddings before detection.
        Matches the preprocessing pipeline in NLP-ADBench.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        Outlier scores of the training data. Higher is more abnormal.

    threshold_ : float
        Score threshold based on ``contamination``.

    labels_ : numpy array of shape (n_samples,)
        Binary labels of training data (0: inlier, 1: outlier).

    encoder_ : BaseEncoder
        The resolved encoder instance.

    detector_ : BaseDetector
        The resolved and fitted detector instance.

    Examples
    --------
    >>> from pyod.models.embedding import EmbeddingOD
    >>> clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN')
    >>> clf.fit(train_texts)
    >>> scores = clf.decision_function(test_texts)
    >>> labels = clf.predict(test_texts)
    """

    def __init__(self, encoder, detector='LUNAR', contamination=0.1,
                 batch_size=32, cache_embeddings=False,
                 reduce_dim=None, standardize=True):
        super(EmbeddingOD, self).__init__(contamination=contamination)
        self.encoder = encoder
        self.detector = detector
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        self.reduce_dim = reduce_dim
        self.standardize = standardize

    def fit(self, X, y=None):
        """Fit detector on raw input data.

        Encodes X into embeddings, applies preprocessing, then fits
        the inner detector.

        Parameters
        ----------
        X : list or array-like
            Raw input data (e.g., list of strings for text,
            list of PIL Images for images).

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.encoder_ = resolve_encoder(self.encoder)
        self.detector_ = resolve_detector(self.detector, self.contamination)

        # Encode
        X_emb = self.encoder_.encode(
            X, batch_size=self.batch_size, show_progress=True)

        # Preprocess (matches NLP-ADBench pipeline)
        X_emb = self._preprocess_fit(X_emb)

        # Cache if requested
        if self.cache_embeddings:
            self.train_embeddings_ = X_emb

        # Fit detector
        self._set_n_classes(y)
        self.detector_.fit(X_emb, y)
        self.decision_scores_ = self.detector_.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly scores for X.

        Parameters
        ----------
        X : list or array-like
            Raw input data in the same format as fit().

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            Anomaly scores. Higher is more abnormal.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        X_emb = self.encoder_.encode(
            X, batch_size=self.batch_size, show_progress=False)
        X_emb = self._preprocess_transform(X_emb)
        return self.detector_.decision_function(X_emb)

    def _preprocess_fit(self, X_emb):
        """Fit preprocessing and transform embeddings."""
        X_emb = np.nan_to_num(X_emb)
        X_emb = np.clip(X_emb, np.finfo(np.float32).min,
                        np.finfo(np.float32).max)

        if self.standardize:
            self.scaler_ = StandardScaler()
            X_emb = self.scaler_.fit_transform(X_emb)

        if self.reduce_dim is not None:
            self.pca_ = PCA(n_components=self.reduce_dim)
            X_emb = self.pca_.fit_transform(X_emb)

        return X_emb.astype(np.float32)

    def _preprocess_transform(self, X_emb):
        """Transform embeddings using fitted preprocessing."""
        X_emb = np.nan_to_num(X_emb)
        X_emb = np.clip(X_emb, np.finfo(np.float32).min,
                        np.finfo(np.float32).max)

        if self.standardize:
            X_emb = self.scaler_.transform(X_emb)

        if self.reduce_dim is not None:
            X_emb = self.pca_.transform(X_emb)

        return X_emb.astype(np.float32)

    # ---- Presets ----

    @classmethod
    def for_text(cls, quality='balanced', **kwargs):
        """Create an EmbeddingOD configured for text anomaly detection.

        Configurations are informed by NLP-ADBench (EMNLP 2025).

        Parameters
        ----------
        quality : str, optional (default='balanced')
            - 'fast': MiniLM encoder (384d) + KNN. No API key needed.
            - 'balanced': mpnet encoder (768d) + LUNAR. No API key needed.
            - 'best': OpenAI large (3072d) + LUNAR. Requires API key.

        **kwargs
            Override any EmbeddingOD parameter.

        Returns
        -------
        clf : EmbeddingOD
        """
        presets = {
            'fast': {
                'encoder': 'all-MiniLM-L6-v2',
                'detector': 'KNN',
            },
            'balanced': {
                'encoder': 'all-mpnet-base-v2',
                'detector': 'LUNAR',
            },
            'best': {
                'encoder': 'text-embedding-3-large',
                'detector': 'LUNAR',
                'cache_embeddings': True,
            },
        }
        if quality not in presets:
            raise ValueError(
                "quality must be 'fast', 'balanced', or 'best', "
                "got '%s'" % quality)
        config = {**presets[quality], **kwargs}
        return cls(**config)

    @classmethod
    def for_image(cls, quality='balanced', **kwargs):
        """Create an EmbeddingOD configured for image anomaly detection.

        Configurations are informed by AnomalyDINO (WACV 2025).

        Parameters
        ----------
        quality : str, optional (default='balanced')
            - 'fast': DINOv2-small (384d) + KNN.
            - 'balanced': DINOv2-base (768d) + LOF.
            - 'best': DINOv2-large (1024d) + KNN.

        **kwargs
            Override any EmbeddingOD parameter.

        Returns
        -------
        clf : EmbeddingOD
        """
        presets = {
            'fast': {
                'encoder': 'dinov2-small',
                'detector': 'KNN',
            },
            'balanced': {
                'encoder': 'dinov2-base',
                'detector': 'LOF',
            },
            'best': {
                'encoder': 'dinov2-large',
                'detector': 'KNN',
            },
        }
        if quality not in presets:
            raise ValueError(
                "quality must be 'fast', 'balanced', or 'best', "
                "got '%s'" % quality)
        config = {**presets[quality], **kwargs}
        return cls(**config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_embedding.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```
git add pyod/models/embedding.py pyod/test/test_embedding.py
git commit -m "feat: add EmbeddingOD wrapper with presets and detector resolution"
```

---

### Task 3: SentenceTransformerEncoder

**Files:**
- Create: `pyod/utils/encoders/sentence_transformer.py`
- Modify: `pyod/test/test_encoders.py`

- [ ] **Step 1: Write the test**

Append to `pyod/test/test_encoders.py`:

```python
import importlib


@unittest.skipUnless(
    importlib.util.find_spec('sentence_transformers') is not None,
    "sentence-transformers not installed")
class TestSentenceTransformerEncoder(unittest.TestCase):
    def setUp(self):
        from pyod.utils.encoders.sentence_transformer import \
            SentenceTransformerEncoder
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
        enc = SentenceTransformerEncoder(
            model_name='all-MiniLM-L6-v2', normalize=True)
        emb = enc.encode(self.texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
```

- [ ] **Step 2: Run test (skip expected if sentence-transformers not installed)**

Run: `python -m pytest pyod/test/test_encoders.py::TestSentenceTransformerEncoder -v`
Expected: SKIP if not installed, otherwise should FAIL (module not found)

- [ ] **Step 3: Implement SentenceTransformerEncoder**

Create `pyod/utils/encoders/sentence_transformer.py`:

```python
# -*- coding: utf-8 -*-
"""SentenceTransformerEncoder for EmbeddingOD."""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from . import BaseEncoder


class SentenceTransformerEncoder(BaseEncoder):
    """Encoder using sentence-transformers library.

    Wraps ``sentence_transformers.SentenceTransformer`` to produce
    text embeddings compatible with PyOD detectors.

    Parameters
    ----------
    model_name : str, optional (default='all-MiniLM-L6-v2')
        Name or path of a sentence-transformers model.

    device : str or None, optional (default=None)
        Device for inference ('cpu', 'cuda', etc.).
        None for auto-detection.

    normalize : bool, optional (default=False)
        L2-normalize output embeddings.

    truncate_dim : int or None, optional (default=None)
        Truncate embeddings to this dimensionality (Matryoshka).

    Examples
    --------
    >>> from pyod.utils.encoders.sentence_transformer import \\
    ...     SentenceTransformerEncoder
    >>> encoder = SentenceTransformerEncoder('all-MiniLM-L6-v2')
    >>> embeddings = encoder.encode(["hello world", "anomaly text"])
    >>> embeddings.shape
    (2, 384)
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', device=None,
                 normalize=False, truncate_dim=None):
        if SentenceTransformer is None:
            raise ImportError(
                "SentenceTransformerEncoder requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers")
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.truncate_dim = truncate_dim

    def encode(self, X, batch_size=32, show_progress=True):
        """Encode text strings to embeddings.

        Parameters
        ----------
        X : list of str
            Text strings to encode.

        batch_size : int, optional (default=32)
            Batch size for encoding.

        show_progress : bool, optional (default=True)
            Show progress bar.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, n_features)
        """
        if not hasattr(self, 'model_'):
            self.model_ = SentenceTransformer(
                self.model_name, device=self.device)

        embeddings = self.model_.encode(
            X,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            truncate_dim=self.truncate_dim,
        )
        return self._validate_output(embeddings)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest pyod/test/test_encoders.py::TestSentenceTransformerEncoder -v`
Expected: PASS if sentence-transformers installed, SKIP otherwise

- [ ] **Step 5: Commit**

```
git add pyod/utils/encoders/sentence_transformer.py pyod/test/test_encoders.py
git commit -m "feat: add SentenceTransformerEncoder"
```

---

### Task 4: OpenAIEncoder

**Files:**
- Create: `pyod/utils/encoders/openai_encoder.py`
- Modify: `pyod/test/test_encoders.py`

- [ ] **Step 1: Write the test**

Append to `pyod/test/test_encoders.py`:

```python
from unittest.mock import patch, MagicMock


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

        # 3000 items should be split into 2 batches (max 2048 per call)
        def side_effect(**kwargs):
            n = len(kwargs['input'])
            return self._make_mock_response(n)

        mock_client.embeddings.create.side_effect = side_effect

        encoder = OpenAIEncoder(model_name='text-embedding-3-small')
        texts = [f"text_{i}" for i in range(3000)]
        emb = encoder.encode(texts)
        assert_equal(emb.shape[0], 3000)
        # Should have been called twice (2048 + 952)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_encoders.py::TestOpenAIEncoder -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement OpenAIEncoder**

Create `pyod/utils/encoders/openai_encoder.py`:

```python
# -*- coding: utf-8 -*-
"""OpenAIEncoder for EmbeddingOD."""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import os

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from . import BaseEncoder

_MAX_BATCH_SIZE = 2048  # OpenAI API limit per request


class OpenAIEncoder(BaseEncoder):
    """Encoder using OpenAI Embeddings API.

    Produces text embeddings via the OpenAI API. Handles batching
    (max 2048 items per request) internally.

    Parameters
    ----------
    model_name : str, optional (default='text-embedding-3-small')
        OpenAI embedding model name.

    dimensions : int or None, optional (default=None)
        Truncate embeddings to this dimensionality (Matryoshka).
        Only supported by text-embedding-3-* models.

    api_key : str or None, optional (default=None)
        OpenAI API key. Falls back to OPENAI_API_KEY environment variable.

    Examples
    --------
    >>> from pyod.utils.encoders.openai_encoder import OpenAIEncoder
    >>> encoder = OpenAIEncoder('text-embedding-3-small')
    >>> embeddings = encoder.encode(["normal text", "anomalous text"])
    """

    def __init__(self, model_name='text-embedding-3-small',
                 dimensions=None, api_key=None):
        if OpenAI is None:
            raise ImportError(
                "OpenAIEncoder requires 'openai'. "
                "Install with: pip install openai")
        self.model_name = model_name
        self.dimensions = dimensions
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')

    def encode(self, X, batch_size=2048, show_progress=True):
        """Encode text strings to embeddings via OpenAI API.

        Parameters
        ----------
        X : list of str
            Text strings to encode.

        batch_size : int, optional (default=2048)
            Batch size. Capped at 2048 (OpenAI API limit).

        show_progress : bool, optional (default=True)
            Show progress bar (not used for API calls).

        Returns
        -------
        embeddings : numpy array of shape (n_samples, n_features)
        """
        if not hasattr(self, 'client_'):
            self.client_ = OpenAI(api_key=self.api_key)

        batch_size = min(batch_size, _MAX_BATCH_SIZE)
        all_embeddings = []

        for i in range(0, len(X), batch_size):
            batch = list(X[i:i + batch_size])
            kwargs = {
                'model': self.model_name,
                'input': batch,
                'encoding_format': 'float',
            }
            if self.dimensions is not None:
                kwargs['dimensions'] = self.dimensions

            response = self.client_.embeddings.create(**kwargs)
            batch_emb = [item.embedding for item in response.data]
            all_embeddings.extend(batch_emb)

        embeddings = np.array(all_embeddings)
        return self._validate_output(embeddings)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest pyod/test/test_encoders.py::TestOpenAIEncoder -v`
Expected: All 3 tests PASS (using mocked API)

- [ ] **Step 5: Commit**

```
git add pyod/utils/encoders/openai_encoder.py pyod/test/test_encoders.py
git commit -m "feat: add OpenAIEncoder with batching"
```

---

### Task 5: HuggingFaceEncoder

**Files:**
- Create: `pyod/utils/encoders/huggingface.py`
- Modify: `pyod/test/test_encoders.py`

- [ ] **Step 1: Write the test**

Append to `pyod/test/test_encoders.py`:

```python
@unittest.skipUnless(
    importlib.util.find_spec('transformers') is not None
    and importlib.util.find_spec('torch') is not None,
    "transformers or torch not installed")
class TestHuggingFaceEncoderText(unittest.TestCase):
    def setUp(self):
        from pyod.utils.encoders.huggingface import HuggingFaceEncoder
        self.encoder = HuggingFaceEncoder(
            model_name='bert-base-uncased', modality='text')
        self.texts = ["Hello world", "Test sentence", "Another one"]

    def test_encode_shape(self):
        emb = self.encoder.encode(self.texts)
        assert_equal(emb.shape[0], 3)
        # BERT base produces 768-dim embeddings
        assert_equal(emb.shape[1], 768)

    def test_encode_dtype(self):
        emb = self.encoder.encode(self.texts)
        assert emb.dtype == np.float64

    def test_encode_cls_pooling(self):
        enc = HuggingFaceEncoder(
            model_name='bert-base-uncased', modality='text', pooling='cls')
        emb = enc.encode(self.texts)
        assert_equal(emb.shape[0], 3)

    def test_encode_mean_pooling(self):
        enc = HuggingFaceEncoder(
            model_name='bert-base-uncased', modality='text', pooling='mean')
        emb = enc.encode(self.texts)
        assert_equal(emb.shape[0], 3)
```

- [ ] **Step 2: Run test (skip expected if transformers/torch not installed)**

Run: `python -m pytest pyod/test/test_encoders.py::TestHuggingFaceEncoderText -v`
Expected: SKIP if not installed, otherwise FAIL

- [ ] **Step 3: Implement HuggingFaceEncoder**

Create `pyod/utils/encoders/huggingface.py`:

```python
# -*- coding: utf-8 -*-
"""HuggingFaceEncoder for EmbeddingOD."""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
except ImportError:
    torch = None
    AutoModel = None

from . import BaseEncoder


class HuggingFaceEncoder(BaseEncoder):
    """Encoder using HuggingFace transformers.

    Supports both text (AutoTokenizer + AutoModel) and image
    (AutoImageProcessor + AutoModel) modalities.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path.

    device : str or None, optional (default=None)
        Device for inference. None for auto-detection.

    pooling : str, optional (default='cls')
        Pooling strategy: 'cls' for CLS token, 'mean' for
        mean of all token embeddings.

    modality : str, optional (default='text')
        Input modality: 'text' or 'image'.

    Examples
    --------
    >>> from pyod.utils.encoders.huggingface import HuggingFaceEncoder
    >>> encoder = HuggingFaceEncoder('bert-base-uncased', modality='text')
    >>> embeddings = encoder.encode(["hello", "world"])
    """

    def __init__(self, model_name, device=None, pooling='cls',
                 modality='text'):
        if AutoModel is None:
            raise ImportError(
                "HuggingFaceEncoder requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch")
        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        self.modality = modality

    def _load_model(self):
        """Load model and processor/tokenizer on first use."""
        if self.device is None:
            if torch.cuda.is_available():
                self.device_ = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and \
                    torch.backends.mps.is_available():
                self.device_ = torch.device('mps')
            else:
                self.device_ = torch.device('cpu')
        else:
            self.device_ = torch.device(self.device)

        self.model_ = AutoModel.from_pretrained(
            self.model_name).to(self.device_)
        self.model_.eval()

        if self.modality == 'image':
            self.processor_ = AutoImageProcessor.from_pretrained(
                self.model_name)
        else:
            self.processor_ = AutoTokenizer.from_pretrained(
                self.model_name)

    def encode(self, X, batch_size=32, show_progress=True):
        """Encode text or images to embeddings.

        Parameters
        ----------
        X : list of str (text) or list of PIL.Image (image)
            Input data.

        batch_size : int, optional (default=32)
            Batch size for encoding.

        show_progress : bool, optional (default=True)
            Show progress bar.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, n_features)
        """
        if not hasattr(self, 'model_'):
            self._load_model()

        all_embeddings = []

        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]

            if self.modality == 'image':
                inputs = self.processor_(
                    images=list(batch), return_tensors='pt'
                ).to(self.device_)
            else:
                inputs = self.processor_(
                    list(batch), return_tensors='pt',
                    padding=True, truncation=True, max_length=512
                ).to(self.device_)

            with torch.no_grad():
                outputs = self.model_(**inputs)

            hidden = outputs.last_hidden_state

            if self.pooling == 'cls':
                emb = hidden[:, 0, :]
            elif self.pooling == 'mean':
                if self.modality == 'text':
                    mask = inputs['attention_mask'].unsqueeze(-1).float()
                    emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    emb = hidden[:, 1:, :].mean(dim=1)
            else:
                raise ValueError(
                    "pooling must be 'cls' or 'mean', got '%s'"
                    % self.pooling)

            all_embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return self._validate_output(embeddings)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest pyod/test/test_encoders.py::TestHuggingFaceEncoderText -v`
Expected: PASS if transformers+torch installed, SKIP otherwise

- [ ] **Step 5: Commit**

```
git add pyod/utils/encoders/huggingface.py pyod/test/test_encoders.py
git commit -m "feat: add HuggingFaceEncoder for text and image"
```

---

### Task 6: Encoder Registry Tests

**Files:**
- Modify: `pyod/test/test_encoders.py`

- [ ] **Step 1: Write registry resolution tests**

Append to `pyod/test/test_encoders.py`:

```python
from pyod.utils.encoders import resolve_encoder, CallableEncoder, \
    BaseEncoder, _ENCODER_REGISTRY


class TestResolveEncoder(unittest.TestCase):
    def test_resolve_callable(self):
        fn = lambda X: np.random.randn(len(X), 5)
        encoder = resolve_encoder(fn)
        assert isinstance(encoder, CallableEncoder)

    def test_resolve_base_encoder_instance(self):
        enc = CallableEncoder(fn=lambda X: np.random.randn(len(X), 5))
        resolved = resolve_encoder(enc)
        assert resolved is enc

    def test_resolve_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            resolve_encoder('definitely-not-a-real-model-xyz-999')

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
        # A full HuggingFace model ID should auto-resolve
        encoder = resolve_encoder(
            'sentence-transformers/all-MiniLM-L6-v2')
        assert hasattr(encoder, 'encode')
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest pyod/test/test_encoders.py::TestResolveEncoder -v`
Expected: All non-skip tests PASS

- [ ] **Step 3: Commit**

```
git add pyod/test/test_encoders.py
git commit -m "test: add encoder registry resolution tests"
```

---

### Task 7: End-to-End Integration Test

**Files:**
- Modify: `pyod/test/test_embedding.py`

- [ ] **Step 1: Write integration test with sentence-transformers**

Append to `pyod/test/test_embedding.py`:

```python
import importlib


@unittest.skipUnless(
    importlib.util.find_spec('sentence_transformers') is not None,
    "sentence-transformers not installed")
class TestEmbeddingODIntegration(unittest.TestCase):
    """End-to-end test with real sentence-transformers encoder."""

    def setUp(self):
        # Normal samples: consistent topic (weather)
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

        # Test set: mix of normal and anomalous
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
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest pyod/test/test_embedding.py::TestEmbeddingODIntegration -v`
Expected: PASS if sentence-transformers installed, SKIP otherwise

- [ ] **Step 3: Commit**

```
git add pyod/test/test_embedding.py
git commit -m "test: add end-to-end integration test for EmbeddingOD"
```

---

### Task 8: Example File and setup.py Extras

**Files:**
- Create: `examples/embedding_od_example.py`
- Modify: `setup.py`

- [ ] **Step 1: Create example file**

Create `examples/embedding_od_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using EmbeddingOD for text anomaly detection.

EmbeddingOD chains a foundation model encoder with any PyOD detector,
enabling anomaly detection on text, image, and other non-tabular data.

This implements the two-step approach shown to outperform end-to-end
methods in NLP-ADBench (Li et al., EMNLP 2025).

Requirements:
    pip install pyod sentence-transformers
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from pyod.models.embedding import EmbeddingOD

# Training data: normal samples (consistent topic)
train_texts = [
    "Quarterly revenue exceeded expectations by 12 percent",
    "The company announced a new product line for Q3",
    "Stock price remained stable after the earnings report",
    "Board of directors approved the annual dividend",
    "Operating costs decreased due to efficiency improvements",
    "Market analysts upgraded the company rating to buy",
    "New partnership expected to drive growth next quarter",
    "Employee headcount grew by 5 percent this year",
] * 20  # 160 training samples

# Test data: mix of normal and anomalous
test_texts = [
    "Annual report shows strong financial performance",    # normal
    "Cost reduction strategy yielded positive results",     # normal
    "The volcano erupted covering the island in ash",       # anomaly
    "Alien signals detected by deep space telescope",       # anomaly
    "Profit margins improved across all business units",    # normal
    "A rare species of deep-sea fish was discovered",       # anomaly
]

# ---- Method 1: Manual configuration ----
print("Method 1: Manual configuration")
clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN',
                  contamination=0.1)
clf.fit(train_texts)

scores = clf.decision_function(test_texts)
labels = clf.predict(test_texts)
proba = clf.predict_proba(test_texts)

for i, text in enumerate(test_texts):
    print(f"  [{labels[i]}] score={scores[i]:.3f}  "
          f"prob={proba[i, 1]:.3f}  {text[:50]}")

# ---- Method 2: Use a preset ----
print("\nMethod 2: Preset (fast text)")
clf2 = EmbeddingOD.for_text(quality='fast')
clf2.fit(train_texts)

labels2 = clf2.predict(test_texts)
for i, text in enumerate(test_texts):
    tag = "ANOMALY" if labels2[i] == 1 else "normal "
    print(f"  {tag}  {text[:50]}")

# ---- Method 3: Custom encoder function ----
print("\nMethod 3: Custom encoder (random projection demo)")
import numpy as np


def hash_encoder(texts):
    """Toy encoder: hash-based random projection."""
    rng = np.random.RandomState(42)
    vocab = {}
    dim = 50
    result = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = rng.randn(dim)
            result[i] += vocab[word]
    return result


clf3 = EmbeddingOD(encoder=hash_encoder, detector='LOF')
clf3.fit(train_texts)
labels3 = clf3.predict(test_texts)
print(f"  Predictions: {labels3}")
```

- [ ] **Step 2: Add optional dependency extras to setup.py**

In `setup.py`, add `extras_require` to the `setup()` call, after the `install_requires` line:

```python
    extras_require={
        'embedding': ['sentence-transformers>=2.0'],
        'openai': ['openai>=1.0'],
        'all': [
            'sentence-transformers>=2.0',
            'openai>=1.0',
            'transformers>=4.0',
            'torch>=2.0',
        ],
    },
```

- [ ] **Step 3: Verify example runs**

Run: `python examples/embedding_od_example.py`
Expected: Output showing anomaly scores and labels for each text. Method 3 (hash encoder) should work without sentence-transformers.

- [ ] **Step 4: Commit**

```
git add examples/embedding_od_example.py setup.py
git commit -m "feat: add EmbeddingOD example and optional dependency extras"
```

---

### Task 9: Full Test Suite Verification

- [ ] **Step 1: Run all new tests together**

Run: `python -m pytest pyod/test/test_encoders.py pyod/test/test_embedding.py -v`
Expected: All tests PASS (some SKIP for optional deps)

- [ ] **Step 2: Run the full PyOD test suite to check for regressions**

Run: `python -m pytest pyod/test/ -v --timeout=300`
Expected: No regressions in existing tests

- [ ] **Step 3: Commit any fixes needed**

If any test fixes are needed, commit them:
```
git add -u
git commit -m "fix: resolve test issues from EmbeddingOD integration"
```

---

## Summary

| Task | What it delivers | Dependencies |
|------|-----------------|--------------|
| 1 | BaseEncoder, CallableEncoder, registry | None (core Python) |
| 2 | EmbeddingOD wrapper, resolve_detector, presets, tests | None (uses CallableEncoder) |
| 3 | SentenceTransformerEncoder | Optional: sentence-transformers |
| 4 | OpenAIEncoder (with mocked tests) | Optional: openai |
| 5 | HuggingFaceEncoder | Optional: transformers, torch |
| 6 | Registry resolution tests | Varies |
| 7 | End-to-end integration test | Optional: sentence-transformers |
| 8 | Example + setup.py extras | None |
| 9 | Full test suite verification | All |

Tasks 1-2 are the core -- everything after that is additive. The system is usable after Task 2 via CallableEncoder.
