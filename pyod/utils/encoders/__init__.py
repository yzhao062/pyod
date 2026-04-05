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
