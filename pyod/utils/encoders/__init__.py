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

    def _validate_output(self, embeddings, n_samples=None):
        """Validate and normalize encoder output.

        Parameters
        ----------
        embeddings : array-like
            Raw encoder output.

        n_samples : int or None
            Expected number of rows. If provided, raises ValueError
            on mismatch.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        check_array(embeddings)
        if n_samples is not None and embeddings.shape[0] != n_samples:
            raise ValueError(
                "Encoder returned %d samples, expected %d"
                % (embeddings.shape[0], n_samples))
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
        return self._validate_output(embeddings, n_samples=len(X))


class MultiModalEncoder(BaseEncoder):
    """Encode multiple modalities and concatenate into a single embedding.

    Each modality is encoded by its own encoder. The resulting embeddings
    are concatenated column-wise into a single feature matrix suitable
    for any PyOD detector.

    Parameters
    ----------
    encoders : dict of {str: encoder}
        Maps modality name to encoder. Each value can be:
        - A string (resolved via resolve_encoder at encode time)
        - A BaseEncoder instance
        - ``'passthrough'`` for pre-computed numeric features

    weights : dict of {str: float} or None, optional (default=None)
        Per-modality scaling applied after encoding. Useful when
        embedding dimensions differ significantly across modalities.

    Examples
    --------
    >>> from pyod.utils.encoders import MultiModalEncoder
    >>> encoder = MultiModalEncoder({
    ...     'text': 'all-MiniLM-L6-v2',
    ...     'tabular': 'passthrough',
    ... })
    >>> data = {'text': ["hello", "world"], 'tabular': np.array([[1, 2], [3, 4]])}
    >>> embeddings = encoder.encode(data)
    >>> embeddings.shape[0]
    2
    """

    def __init__(self, encoders, weights=None):
        if not isinstance(encoders, dict) or len(encoders) == 0:
            raise ValueError("encoders must be a non-empty dict")
        self.encoders = encoders
        self.weights = weights

    def fit_encode(self, X, batch_size=32, show_progress=True):
        """Encode training data and store per-modality mean embeddings.

        Call this during training (EmbeddingOD.fit) so that mean
        embeddings are available for imputing missing samples at
        test time. Subsequent calls to ``encode`` will use these
        stored means.

        Parameters
        ----------
        X : dict of {str: data}
            Training data. Should not contain ``None`` samples.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, total_features)
        """
        emb = self.encode(X, batch_size=batch_size,
                          show_progress=show_progress)
        # Store per-modality means from training for imputation.
        # Use _last_parts_unweighted_ (before weights) so that
        # weights are applied exactly once during encode.
        self.means_ = {}
        for name, part in self._last_parts_unweighted_.items():
            self.means_[name] = np.mean(part, axis=0)
        return emb

    def encode(self, X, batch_size=32, show_progress=True):
        """Encode multi-modal input and concatenate.

        Parameters
        ----------
        X : dict of {str: data}
            Maps modality name to input data. Keys must match
            the ``encoders`` dict. Individual samples may be ``None``
            to indicate a missing modality for that sample; missing
            embeddings are imputed with the training mean (if
            ``fit_encode`` was called) or zeros.

        batch_size : int, optional (default=32)
            Batch size for encoding.

        show_progress : bool, optional (default=True)
            Show progress bar.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, total_features)
        """
        if not isinstance(X, dict):
            raise TypeError(
                "MultiModalEncoder expects a dict input, got %s" % type(X))

        # Resolve encoders on first call
        if not hasattr(self, 'resolved_'):
            self.resolved_ = {}
            for name, enc in self.encoders.items():
                if enc == 'passthrough':
                    self.resolved_[name] = 'passthrough'
                else:
                    self.resolved_[name] = resolve_encoder(enc)

        parts = []
        self._last_parts_ = {}
        self._last_parts_unweighted_ = {}
        n_samples = None
        for name, enc in self.resolved_.items():
            if name not in X:
                raise KeyError(
                    "Modality '%s' not found in input. "
                    "Expected keys: %s" % (name, list(self.resolved_.keys())))

            modality_data = X[name]
            has_missing = isinstance(modality_data, (list, tuple)) and \
                any(v is None for v in modality_data)

            if has_missing:
                present_idx = [i for i, v in enumerate(modality_data)
                               if v is not None]
                if len(present_idx) == 0:
                    raise ValueError(
                        "All samples are None for modality '%s'" % name)

                # Encode or passthrough the present samples
                if enc == 'passthrough':
                    present_vals = [modality_data[i] for i in present_idx]
                    first = np.asarray(present_vals[0], dtype=np.float64)
                    n_feat = 1 if first.ndim == 0 else first.shape[0]
                    present_emb = np.zeros((len(present_idx), n_feat),
                                           dtype=np.float64)
                    for j, idx in enumerate(present_idx):
                        present_emb[j] = np.asarray(
                            modality_data[idx], dtype=np.float64)
                else:
                    present_data = [modality_data[i] for i in present_idx]
                    present_emb = enc.encode(present_data,
                                             batch_size=batch_size,
                                             show_progress=show_progress)

                # Impute missing with training mean or zeros
                n_total = len(modality_data)
                fill = self.means_.get(name, np.zeros(present_emb.shape[1])) \
                    if hasattr(self, 'means_') else np.zeros(present_emb.shape[1])
                emb = np.tile(fill, (n_total, 1))
                for j, idx in enumerate(present_idx):
                    emb[idx] = present_emb[j]
            elif enc == 'passthrough':
                emb = np.asarray(modality_data, dtype=np.float64)
                if emb.ndim == 1:
                    emb = emb.reshape(-1, 1)
            else:
                emb = enc.encode(modality_data, batch_size=batch_size,
                                 show_progress=show_progress)

            if n_samples is None:
                n_samples = emb.shape[0]
            elif emb.shape[0] != n_samples:
                raise ValueError(
                    "Modality '%s' has %d samples, expected %d"
                    % (name, emb.shape[0], n_samples))

            self._last_parts_unweighted_[name] = emb.copy()

            if self.weights is not None and name in self.weights:
                emb = emb * self.weights[name]

            self._last_parts_[name] = emb
            parts.append(emb)

        return self._validate_output(np.hstack(parts), n_samples=n_samples)


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


_INSTALL_HINTS = {
    'sentence_transformer': 'pip install sentence-transformers',
    'openai': 'pip install openai',
    'huggingface': 'pip install transformers torch',
}


def _create_encoder(backend, **kwargs):
    """Create an encoder from a backend name and kwargs."""
    module_path, class_name = _ENCODER_BACKENDS[backend]
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        hint = _INSTALL_HINTS.get(backend, '')
        raise ImportError(
            "Encoder backend '%s' requires module '%s' which is not "
            "installed. Install with: %s" % (backend, module_path, hint))
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
          The auto-resolve fallback is designed for text embedding models.
          For image models (DINOv2, CLIP, etc.), use registry shortcuts
          (e.g., 'dinov2-small', 'clip-vit-base') instead of raw
          HuggingFace model IDs.

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
        except ImportError:
            pass

        try:
            return _create_encoder('huggingface',
                                   model_name=encoder,
                                   modality='text')
        except ImportError:
            pass

        raise ValueError(
            "Cannot resolve encoder '%s'. Provide a registry shortcut "
            "(e.g., 'all-MiniLM-L6-v2'), a HuggingFace model ID, a "
            "BaseEncoder instance, or a callable." % encoder)

    raise TypeError("encoder must be str, BaseEncoder, or callable, "
                    "got %s" % type(encoder))
