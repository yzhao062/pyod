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
from scipy.special import erf
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    # DeepSVDD requires n_features as a positional arg; cannot resolve
    # from string alone. Pass a configured instance instead.
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
        return clone(detector)

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

    # ---- Overrides for list-based X ----

    def predict_proba(self, X, method='linear', return_confidence=False):
        """Predict the probability of a sample being an outlier.

        Overrides the base implementation to handle list inputs (raw data
        such as text or images) which do not have a ``.shape`` attribute.

        Parameters
        ----------
        X : list or array-like
            Raw input data in the same format as fit().

        method : str, optional (default='linear')
            Probability conversion method. One of 'linear' or 'unify'.

        return_confidence : boolean, optional (default=False)
            If True, also return the confidence of prediction.

        Returns
        -------
        outlier_probability : numpy array of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_
        test_scores = self.decision_function(X)
        n_samples = len(X)

        probs = np.zeros([n_samples, int(self._classes)])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            if return_confidence:
                confidence = self.predict_confidence(X)
                return probs, confidence
            return probs

        elif method == 'unify':
            pre_erf_score = (test_scores - self._mu) / (
                self._sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            if return_confidence:
                confidence = self.predict_confidence(X)
                return probs, confidence
            return probs

        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')

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
