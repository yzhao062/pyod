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
        return self._validate_output(embeddings, n_samples=len(X))
