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
        return self._validate_output(embeddings, n_samples=len(X))
