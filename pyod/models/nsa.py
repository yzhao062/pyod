# -*- coding: utf-8 -*-
"""Negative Selection Algorithm (NSA) family for PyOD.

This module provides PyOD-compatible implementations of the major NSA
families described in the NSA review literature: binary/string NSA,
real-valued NSA, V-detector, grid/matrix NSA, adaptive/dynamic NSA,
optimization-based NSA, clustering-based NSA, density/antigen NSA,
ensemble NSA, boundary-aware NSA, and online/feedback variants.

The classes intentionally follow PyOD conventions:

* ``fit(X, y=None)`` stores ``decision_scores_`` on the training set.
* ``decision_function(X)`` returns larger scores for more abnormal samples.
* ``_process_decision_scores()`` derives ``threshold_`` and ``labels_``.

Several named subclasses are faithful engineering approximations of the
published NSA families rather than line-by-line reproductions of each cited
paper. They expose the same operational idea in a single PyOD detector API.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


_REAL_VARIANTS = {
    "rnsa",
    "rrnsa",
    "vdetector",
    "ansa",
    "evoseedrnsa",
    "ornsa",
    "optimized_nsa",
    "ftnsa",
    "ivrnsa",
    "cb_nsa",
    "prr2nsa",
    "nsa_de",
    "hnsa_idsa",
    "nsa_pso",
    "io_rnsa",
    "biorv_nsa",
    "nsa_ii",
    "oalfb_nsa",
    "fb_nsa",
    "densa",
    "antigen_nsa",
    "nsnad",
    "ren",
    "ainsa",
    "odnsa",
    "cnsa",
    "vor_nsa",
}

_GRID_VARIANTS = {"gnsa", "gf_rnsa", "matrix_nsa"}
_BINARY_VARIANTS = {"bnsa", "binary_nsa"}
_ENSEMBLE_VARIANTS = {"mnsa"}


@dataclass
class _DetectorSet:
    """Container for real-valued NSA detector centers and radii."""

    centers: np.ndarray
    radii: np.ndarray


def _normalise_variant(variant: str) -> str:
    return str(variant).lower().replace("-", "_").replace(" ", "_")


def _safe_scale(values: np.ndarray) -> float:
    scale = float(np.percentile(values, 95)) if values.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return scale


def _pairwise_euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between all rows of X and Y."""
    if Y.size == 0:
        return np.empty((X.shape[0], 0))
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _hamming_matrix(X_bits: np.ndarray, Y_bits: np.ndarray) -> np.ndarray:
    if Y_bits.size == 0:
        return np.empty((X_bits.shape[0], 0), dtype=float)
    return np.mean(X_bits[:, None, :] != Y_bits[None, :, :], axis=2)


class NegativeSelection(BaseDetector):
    """Unified Negative Selection Algorithm family detector.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        Expected proportion of outliers in the data set.

    variant : str, optional (default='vdetector')
        NSA family variant. Supported values include ``'bnsa'``, ``'rnsa'``,
        ``'rrnsa'``, ``'vdetector'``, ``'gnsa'``, ``'gf_rnsa'``,
        ``'matrix_nsa'``, ``'ansa'``, ``'evoseedrnsa'``, ``'ornsa'``,
        ``'optimized_nsa'``, ``'ftnsa'``, ``'ivrnsa'``, ``'cb_nsa'``,
        ``'prr2nsa'``, ``'nsa_de'``, ``'hnsa_idsa'``, ``'nsa_pso'``,
        ``'io_rnsa'``, ``'biorv_nsa'``, ``'nsa_ii'``, ``'oalfb_nsa'``,
        ``'fb_nsa'``, ``'densa'``, ``'mnsa'``, ``'antigen_nsa'``,
        ``'nsnad'``, ``'ren'``, ``'ainsa'``, ``'odnsa'``, ``'cnsa'``, and
        ``'vor_nsa'``.

    n_detectors : int, optional (default=128)
        Number of detectors to retain for detector-based variants.

    radius : float or None, optional (default=None)
        Fixed detector radius in scaled feature space. If ``None``, a radius
        is estimated from the self-nearest-neighbor distances.

    n_candidates : int or None, optional (default=None)
        Candidate pool size before pruning. Defaults to ``max(10*n_detectors,
        512)``.

    sampling_margin : float, optional (default=0.50)
        Expansion around the scaled training bounds for candidate generation.
        A positive margin lets detectors cover near-outside non-self regions.

    n_grid : int, optional (default=8)
        Number of bins per selected feature for grid/matrix variants.

    n_bits : int, optional (default=32)
        Number of binary features for binary/string NSA. Continuous inputs are
        converted into bit strings through quantile thresholds.

    r : int, optional (default=4)
        Matching threshold for binary NSA. For ``match_rule='hamming'``, ``r``
        is interpreted as the maximum number of bit mismatches. For
        ``match_rule='rcb'`` or ``'rchunk'``, it controls contiguous/chunk
        matching length.

    match_rule : {'hamming', 'rcb', 'rchunk'}, optional (default='hamming')
        Binary NSA matching rule.

    n_clusters : int, optional (default=12)
        Number of self clusters for clustering-based variants.

    n_estimators : int, optional (default=5)
        Number of subdetectors for MNSA / ensemble variants.

    feature_subsample : float, optional (default=1.0)
        Fraction of features to retain. Used explicitly by NSNAD and can also
        be used for high-dimensional data.

    optimization_iter : int, optional (default=15)
        Number of lightweight optimization iterations for PSO/DE/optimized
        variants.

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for reproducibility.

    novelty_score_weight : float, optional (default=0.35)
        Weight of the nearest-self-distance fallback used to reduce uncovered
        non-self holes. Higher values make the model behave more like a
        one-class distance detector; lower values emphasize detector hits.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        Outlier scores of the training data. Higher values are more abnormal.

    threshold_ : float
        Threshold derived by PyOD from ``decision_scores_`` and
        ``contamination``.

    labels_ : numpy array of shape (n_samples,)
        Binary labels for the training data.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        variant: str = "vdetector",
        n_detectors: int = 128,
        radius: Optional[float] = None,
        n_candidates: Optional[int] = None,
        sampling_margin: float = 0.50,
        n_grid: int = 8,
        n_bits: int = 32,
        r: int = 4,
        match_rule: str = "hamming",
        n_clusters: int = 12,
        n_estimators: int = 5,
        feature_subsample: float = 1.0,
        optimization_iter: int = 15,
        random_state: Optional[int] = None,
        novelty_score_weight: float = 0.35,
    ) -> None:
        super(NegativeSelection, self).__init__(contamination=contamination)
        self.variant = variant
        self.n_detectors = n_detectors
        self.radius = radius
        self.n_candidates = n_candidates
        self.sampling_margin = sampling_margin
        self.n_grid = n_grid
        self.n_bits = n_bits
        self.r = r
        self.match_rule = match_rule
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.feature_subsample = feature_subsample
        self.optimization_iter = optimization_iter
        self.random_state = random_state
        self.novelty_score_weight = novelty_score_weight

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        """Fit the selected NSA variant.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Training samples. In the standard one-class setting, these are
            self/normal examples. Semi-supervised variants can use ``y`` if it
            is supplied with 0 for self and 1 for non-self.

        y : array-like of shape (n_samples,), optional
            Optional labels for semi-supervised variants such as HNSA-IDSA and
            NSA-II. This parameter is otherwise ignored for PyOD consistency.
        """
        X = check_array(X)
        self._set_n_classes(y)
        self._variant_ = _normalise_variant(self.variant)
        if self._variant_ not in (_REAL_VARIANTS | _GRID_VARIANTS | _BINARY_VARIANTS | _ENSEMBLE_VARIANTS):
            raise ValueError("Unsupported NSA variant: %r" % self.variant)

        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features_in_ = X_scaled.shape[1]
        self.feature_indices_ = self._select_features(X_scaled)
        X_selected = X_scaled[:, self.feature_indices_]

        if self._variant_ in {"ornsa", "ftnsa"}:
            X_selected = self._robust_self_subset(X_selected)
        self.self_X_ = X_selected

        if self._variant_ in _ENSEMBLE_VARIANTS:
            self._fit_ensemble(X, y)
            self.decision_scores_ = self.decision_function(X)
        elif self._variant_ in _BINARY_VARIANTS:
            self._fit_binary(X_scaled)
            self.decision_scores_ = self._score_binary_training(X_scaled)
        elif self._variant_ in _GRID_VARIANTS:
            self._fit_grid(X_selected)
            self.decision_scores_ = self._score_grid(X_selected)
        else:
            self._fit_real(X_selected, y)
            self.decision_scores_ = self._score_real_training(X_selected)

        self._process_decision_scores()
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return raw NSA anomaly scores.

        Higher scores indicate more abnormal samples.
        """
        check_is_fitted(self, ["scaler_", "feature_indices_"])
        X = check_array(X)
        X_scaled = self.scaler_.transform(X)
        X_selected = X_scaled[:, self.feature_indices_]

        if self._variant_ in _ENSEMBLE_VARIANTS:
            return self._score_ensemble(X)
        if self._variant_ in _BINARY_VARIANTS:
            return self._score_binary(X_scaled)
        if self._variant_ in _GRID_VARIANTS:
            return self._score_grid(X_selected)
        return self._score_real(X_selected)

    def partial_fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        """Update online/feedback variants with new self samples.

        For non-online variants this method refits the detector on the combined
        old and new self profile. OALFB-NSA and FB-NSA use this as their online
        feedback mechanism.
        """
        check_is_fitted(self, ["X_train_original_"])
        X = check_array(X)
        X_all = np.vstack([self.X_train_original_, X])
        return self.fit(X_all, y=None)

    def _select_features(self, X_scaled: np.ndarray) -> np.ndarray:
        n_features = X_scaled.shape[1]
        if self._variant_ == "nsnad" or self.feature_subsample < 1.0:
            keep = max(1, int(np.ceil(n_features * min(max(self.feature_subsample, 0.01), 1.0))))
            variances = np.var(X_scaled, axis=0)
            return np.argsort(variances)[-keep:]
        return np.arange(n_features)

    def _robust_self_subset(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] < 12:
            return X
        nn = NearestNeighbors(n_neighbors=min(6, X.shape[0])).fit(X)
        dists, _ = nn.kneighbors(X)
        score = np.mean(dists[:, 1:], axis=1)
        cutoff = np.percentile(score, 90)
        subset = X[score <= cutoff]
        return subset if subset.shape[0] >= 5 else X

    def _fit_ensemble(self, X: np.ndarray, y: Optional[ArrayLike]) -> None:
        variants = ["rnsa", "vdetector", "gnsa", "ansa", "densa"]
        self.ensemble_detectors_: List[NegativeSelection] = []
        n_estimators = max(1, int(self.n_estimators))
        for i in range(n_estimators):
            v = variants[i % len(variants)]
            model = NegativeSelection(
                contamination=self.contamination if isinstance(self.contamination, (float, int)) else 0.1,
                variant=v,
                n_detectors=max(16, self.n_detectors // max(1, n_estimators // 2)),
                radius=self.radius,
                n_candidates=self.n_candidates,
                sampling_margin=self.sampling_margin,
                n_grid=self.n_grid,
                n_bits=self.n_bits,
                r=self.r,
                match_rule=self.match_rule,
                n_clusters=self.n_clusters,
                n_estimators=1,
                feature_subsample=self.feature_subsample,
                optimization_iter=max(3, self.optimization_iter // 2),
                random_state=None if self.random_state is None else int(self.random_state) + i + 1,
                novelty_score_weight=self.novelty_score_weight,
            )
            model.fit(X, y)
            self.ensemble_detectors_.append(model)
        self.X_train_original_ = X.copy()

    def _fit_binary(self, X_scaled: np.ndarray) -> None:
        self.X_train_original_ = self.scaler_.inverse_transform(X_scaled).copy()
        thresholds = []
        for j in range(X_scaled.shape[1]):
            qs = np.linspace(0.0, 1.0, max(2, int(np.ceil(self.n_bits / X_scaled.shape[1])) + 2))[1:-1]
            thresholds.extend([(j, q, np.quantile(X_scaled[:, j], q)) for q in qs])
        if len(thresholds) == 0:
            thresholds = [(0, 0.5, np.quantile(X_scaled[:, 0], 0.5))]
        thresholds = thresholds[: self.n_bits]
        self.binary_thresholds_ = thresholds
        self.self_bits_ = self._binarize(X_scaled)

        rng = np.random.RandomState(self.random_state)
        n_bits = self.self_bits_.shape[1]
        n_candidates = self.n_candidates or max(10 * self.n_detectors, 512)
        candidates = rng.randint(0, 2, size=(n_candidates, n_bits)).astype(bool)
        valid = []
        for cand in candidates:
            if not self._binary_matches_self(cand):
                valid.append(cand)
            if len(valid) >= self.n_detectors:
                break
        if len(valid) == 0:
            valid = candidates[: min(self.n_detectors, len(candidates))]
        self.binary_detectors_ = np.asarray(valid, dtype=bool)
        self.self_distance_scale_ = max(1.0 / max(n_bits, 1), 1e-12)

    def _binarize(self, X_scaled: np.ndarray) -> np.ndarray:
        bits = []
        for feature, _, threshold in self.binary_thresholds_:
            bits.append(X_scaled[:, feature] > threshold)
        return np.asarray(bits, dtype=bool).T

    def _binary_matches_self(self, detector_bits: np.ndarray) -> bool:
        if self.match_rule == "hamming":
            return np.any(np.sum(self.self_bits_ != detector_bits, axis=1) <= self.r)
        if self.match_rule == "rcb":
            return np.any([self._has_contiguous_match(row, detector_bits, self.r) for row in self.self_bits_])
        if self.match_rule == "rchunk":
            r = max(1, self.r)
            for row in self.self_bits_:
                for start in range(0, len(row), r):
                    if np.array_equal(row[start : start + r], detector_bits[start : start + r]):
                        return True
            return False
        raise ValueError("Unsupported match_rule: %r" % self.match_rule)

    @staticmethod
    def _has_contiguous_match(a: np.ndarray, b: np.ndarray, r: int) -> bool:
        r = max(1, int(r))
        if r > len(a):
            return np.array_equal(a, b)
        equal = a == b
        run = 0
        for value in equal:
            run = run + 1 if value else 0
            if run >= r:
                return True
        return False


    def _score_binary_training(self, X_scaled: np.ndarray) -> np.ndarray:
        X_bits = self._binarize(X_scaled)
        det_dist = _hamming_matrix(X_bits, self.binary_detectors_)
        if det_dist.shape[1]:
            detector_score = np.maximum(0.0, 1.0 - det_dist / max(self.r / X_bits.shape[1], 1.0 / X_bits.shape[1])).max(axis=1)
        else:
            detector_score = np.zeros(X_bits.shape[0])
        self_dist = _hamming_matrix(X_bits, self.self_bits_)
        if self_dist.shape[1] > 1:
            np.fill_diagonal(self_dist, np.inf)
        novelty = np.min(self_dist, axis=1) / max(self.self_distance_scale_, 1e-12)
        novelty[~np.isfinite(novelty)] = 0.0
        return detector_score + self.novelty_score_weight * novelty

    def _score_binary(self, X_scaled: np.ndarray) -> np.ndarray:
        X_bits = self._binarize(X_scaled)
        det_dist = _hamming_matrix(X_bits, self.binary_detectors_)
        if det_dist.shape[1]:
            detector_score = np.maximum(0.0, 1.0 - det_dist / max(self.r / X_bits.shape[1], 1.0 / X_bits.shape[1])).max(axis=1)
        else:
            detector_score = np.zeros(X_bits.shape[0])
        self_dist = _hamming_matrix(X_bits, self.self_bits_)
        novelty = self_dist.min(axis=1) / max(self.self_distance_scale_, 1e-12)
        return detector_score + self.novelty_score_weight * novelty

    def _fit_grid(self, X: np.ndarray) -> None:
        self.X_train_original_ = self.scaler_.inverse_transform(
            self._restore_feature_matrix(X, self.n_features_in_)
        ).copy() if X.shape[1] == self.n_features_in_ else np.empty((0, self.n_features_in_))
        self.grid_min_ = np.min(X, axis=0) - self.sampling_margin
        self.grid_max_ = np.max(X, axis=0) + self.sampling_margin
        span = np.maximum(self.grid_max_ - self.grid_min_, 1e-12)
        self.grid_edges_ = [np.linspace(self.grid_min_[j], self.grid_max_[j], self.n_grid + 1) for j in range(X.shape[1])]
        self.self_cells_ = self._grid_cells(X)
        self.unique_self_cells_ = np.unique(self.self_cells_, axis=0)
        self.self_cell_set_ = {tuple(row) for row in self.unique_self_cells_}

        rng = np.random.RandomState(self.random_state)
        n_candidates = self.n_candidates or max(10 * self.n_detectors, 512)
        cand = rng.randint(0, self.n_grid, size=(n_candidates, X.shape[1]))
        seen = set()
        detectors = []
        for row in cand:
            key = tuple(row)
            if key not in self.self_cell_set_ and key not in seen:
                detectors.append(row)
                seen.add(key)
            if len(detectors) >= self.n_detectors:
                break
        if len(detectors) == 0:
            detectors = cand[: min(self.n_detectors, len(cand))]
        self.grid_detectors_ = np.asarray(detectors, dtype=int)
        self.self_distance_scale_ = _safe_scale(self._grid_nearest_self_distance(self.self_cells_))

    def _restore_feature_matrix(self, X_selected: np.ndarray, n_features: int) -> np.ndarray:
        X_restored = np.zeros((X_selected.shape[0], n_features))
        X_restored[:, self.feature_indices_] = X_selected
        return X_restored

    def _grid_cells(self, X: np.ndarray) -> np.ndarray:
        cells = []
        for j, edges in enumerate(self.grid_edges_):
            idx = np.digitize(X[:, j], edges[1:-1], right=False)
            idx = np.clip(idx, 0, self.n_grid - 1)
            cells.append(idx)
        return np.asarray(cells, dtype=int).T

    def _grid_nearest_self_distance(self, cells: np.ndarray) -> np.ndarray:
        if self.unique_self_cells_.size == 0:
            return np.zeros(cells.shape[0])
        d = _pairwise_euclidean(cells.astype(float), self.unique_self_cells_.astype(float))
        return d.min(axis=1)

    def _score_grid(self, X: np.ndarray) -> np.ndarray:
        cells = self._grid_cells(X)
        keys = [tuple(row) for row in cells]
        unseen = np.asarray([0.0 if key in self.self_cell_set_ else 1.0 for key in keys])
        det_d = _pairwise_euclidean(cells.astype(float), self.grid_detectors_.astype(float))
        detector_score = np.zeros(cells.shape[0])
        if det_d.shape[1]:
            detector_score = 1.0 / (1.0 + det_d.min(axis=1))
        novelty = self._grid_nearest_self_distance(cells) / max(self.self_distance_scale_, 1e-12)
        return unseen + detector_score + self.novelty_score_weight * novelty

    def _fit_real(self, X: np.ndarray, y: Optional[ArrayLike]) -> None:
        self.X_train_original_ = self.scaler_.inverse_transform(self._restore_feature_matrix(X, self.n_features_in_)).copy()
        self.self_nn_ = NearestNeighbors(n_neighbors=1).fit(X)
        self.self_knn_ = NearestNeighbors(n_neighbors=min(6, X.shape[0])).fit(X)
        self.self_distance_scale_ = self._estimate_self_distance_scale(X)
        self.default_radius_ = self._estimate_default_radius(X)

        if self._variant_ in {"hnsa_idsa", "nsa_ii"} and y is not None:
            y_arr = np.asarray(y)
            if np.any(y_arr == 1):
                X_all = self.scaler_.transform(check_array(self.X_train_original_))[:, self.feature_indices_]
                nonself = X_all[y_arr == 1]
                centers = nonself[: self.n_detectors]
                radii = np.full(centers.shape[0], self.default_radius_)
                self.detector_set_ = _DetectorSet(centers=centers, radii=radii)
                return

        if self._variant_ in {"cb_nsa", "prr2nsa", "cnsa"}:
            X_for_generation = self._cluster_self(X)
        else:
            X_for_generation = X

        candidates = self._candidate_pool(X_for_generation)
        candidates = self._optimise_candidates(candidates, X_for_generation)
        centers, radii = self._mature_detectors(candidates, X_for_generation)
        if centers.shape[0] == 0:
            centers = candidates[: min(self.n_detectors, len(candidates))]
            radii = np.full(centers.shape[0], max(self.default_radius_, 1e-3))
        self.detector_set_ = _DetectorSet(centers=centers, radii=radii)

    def _estimate_self_distance_scale(self, X: np.ndarray) -> float:
        if X.shape[0] < 2:
            return 1.0
        nn = NearestNeighbors(n_neighbors=min(2, X.shape[0])).fit(X)
        dists, _ = nn.kneighbors(X)
        values = dists[:, -1]
        return _safe_scale(values)

    def _estimate_default_radius(self, X: np.ndarray) -> float:
        if self.radius is not None:
            return float(self.radius)
        if X.shape[0] < 2:
            return 0.1
        nn = NearestNeighbors(n_neighbors=min(2, X.shape[0])).fit(X)
        dists, _ = nn.kneighbors(X)
        base = float(np.percentile(dists[:, -1], 90))
        if not np.isfinite(base) or base <= 1e-12:
            base = 0.05
        return base

    def _cluster_self(self, X: np.ndarray) -> np.ndarray:
        n_clusters = min(max(2, int(self.n_clusters)), max(2, X.shape[0] // 2))
        if X.shape[0] <= n_clusters:
            return X
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        km.fit(X)
        self.cluster_centers_ = km.cluster_centers_
        return km.cluster_centers_

    def _candidate_pool(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n_candidates = int(self.n_candidates or max(10 * self.n_detectors, 512))
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        low = mins - self.sampling_margin * span
        high = maxs + self.sampling_margin * span

        if self._variant_ in {"densa", "antigen_nsa"}:
            # Bias candidate generation toward low-density regions by starting
            # from points far from random self anchors.
            cand = rng.uniform(low=low, high=high, size=(n_candidates * 2, X.shape[1]))
            d = self.self_nn_.kneighbors(cand, return_distance=True)[0].ravel()
            keep = np.argsort(d)[-n_candidates:]
            return cand[keep]

        if self._variant_ in {"io_rnsa", "ainsa"}:
            levels = []
            for factor in np.linspace(0.25, 1.0 + self.sampling_margin, 4):
                levels.append(rng.uniform(mins - factor * span, maxs + factor * span, size=(n_candidates // 4 + 1, X.shape[1])))
            return np.vstack(levels)[:n_candidates]

        if self._variant_ == "vor_nsa":
            # Vectorized / big-data version. Candidate generation is intentionally
            # batch-oriented to avoid Python loops.
            return rng.uniform(low=low, high=high, size=(n_candidates, X.shape[1]))

        return rng.uniform(low=low, high=high, size=(n_candidates, X.shape[1]))

    def _fitness(self, candidates: np.ndarray, X: np.ndarray) -> np.ndarray:
        d_self = self.self_nn_.kneighbors(candidates, return_distance=True)[0].ravel()
        # Penalize candidate concentration by distance to nearest other candidate.
        if candidates.shape[0] > 1:
            nn = NearestNeighbors(n_neighbors=min(2, candidates.shape[0])).fit(candidates)
            d_cand = nn.kneighbors(candidates, return_distance=True)[0][:, -1]
        else:
            d_cand = np.ones(candidates.shape[0])
        return d_self + 0.15 * d_cand

    def _optimise_candidates(self, candidates: np.ndarray, X: np.ndarray) -> np.ndarray:
        variant = self._variant_
        if variant not in {
            "evoseedrnsa",
            "optimized_nsa",
            "nsa_de",
            "nsa_pso",
            "io_rnsa",
            "ainsa",
            "odnsa",
            "cnsa",
        }:
            return candidates

        rng = np.random.RandomState(self.random_state)
        best = candidates.copy()
        mins = np.min(X, axis=0) - self.sampling_margin
        maxs = np.max(X, axis=0) + self.sampling_margin

        for _ in range(max(1, int(self.optimization_iter))):
            if variant == "nsa_pso":
                nearest_idx = self.self_nn_.kneighbors(best, return_distance=False).ravel()
                away = best - X[nearest_idx % X.shape[0]]
                norm = np.linalg.norm(away, axis=1, keepdims=True) + 1e-12
                step = 0.05 * away / norm + rng.normal(0, 0.015, size=best.shape)
                proposal = np.clip(best + step, mins, maxs)
            elif variant == "nsa_de":
                idx = rng.randint(0, best.shape[0], size=(best.shape[0], 3))
                proposal = best[idx[:, 0]] + 0.5 * (best[idx[:, 1]] - best[idx[:, 2]])
                proposal = np.clip(proposal, mins, maxs)
            else:
                proposal = best + rng.normal(0, 0.04, size=best.shape)
                proposal = np.clip(proposal, mins, maxs)

            replace = self._fitness(proposal, X) > self._fitness(best, X)
            best[replace] = proposal[replace]

        order = np.argsort(self._fitness(best, X))[::-1]
        return best[order]

    def _mature_detectors(self, candidates: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_self = self.self_nn_.kneighbors(candidates, return_distance=True)[0].ravel()
        base_radius = self.default_radius_
        centers: List[np.ndarray] = []
        radii: List[float] = []

        order = np.argsort(d_self)[::-1]
        for idx in order:
            cand = candidates[idx]
            if self._variant_ in {"rnsa", "rrnsa", "ftnsa", "vor_nsa"}:
                rad = min(base_radius, max(d_self[idx] - 1e-6, 0.0))
            elif self._variant_ in {"odnsa", "ren"}:
                rad = max(min(d_self[idx] * 0.85, 2.0 * base_radius), 1e-6)
            elif self._variant_ in {"ansa", "densa", "antigen_nsa"}:
                local_density_radius = self._local_density_radius(cand)
                rad = max(min(d_self[idx] - 1e-6, local_density_radius), 1e-6)
            else:
                rad = max(d_self[idx] - 1e-6, 1e-6)

            if rad <= 1e-8:
                continue
            if centers and self._variant_ in {"biorv_nsa", "ivrnsa", "ren", "optimized_nsa", "odnsa"}:
                d_existing = np.linalg.norm(np.asarray(centers) - cand, axis=1)
                # Self-inhibition/reduced-overlap rule.
                if np.any(d_existing < (np.asarray(radii) + rad) * 0.5):
                    continue
            centers.append(cand)
            radii.append(float(rad))
            if len(centers) >= self.n_detectors:
                break

        if not centers:
            return np.empty((0, X.shape[1])), np.empty((0,))
        return np.vstack(centers), np.asarray(radii)

    def _local_density_radius(self, candidate: np.ndarray) -> float:
        dists, _ = self.self_knn_.kneighbors(candidate.reshape(1, -1))
        density_scale = float(np.mean(dists))
        if not np.isfinite(density_scale) or density_scale <= 1e-12:
            density_scale = self.default_radius_
        return density_scale


    def _score_real_training(self, X: np.ndarray) -> np.ndarray:
        detectors = self.detector_set_
        if X.shape[0] > 1:
            nn = NearestNeighbors(n_neighbors=min(2, X.shape[0])).fit(X)
            d_self = nn.kneighbors(X, return_distance=True)[0][:, -1]
        else:
            d_self = np.zeros(X.shape[0])
        novelty = d_self / max(self.self_distance_scale_, 1e-12)
        if detectors.centers.size == 0:
            return self.novelty_score_weight * novelty
        d_det = _pairwise_euclidean(X, detectors.centers)
        margin = detectors.radii.reshape(1, -1) - d_det
        denom = np.maximum(detectors.radii.reshape(1, -1), 1e-12)
        detector_hit_score = np.maximum(margin / denom, 0.0).max(axis=1)
        if self._variant_ in {"densa", "antigen_nsa"}:
            return detector_hit_score + 0.60 * novelty
        if self._variant_ in {"hnsa_idsa", "nsa_ii"}:
            return detector_hit_score + 0.45 * novelty
        return detector_hit_score + self.novelty_score_weight * novelty

    def _score_real(self, X: np.ndarray) -> np.ndarray:
        detectors = self.detector_set_
        d_self = self.self_nn_.kneighbors(X, return_distance=True)[0].ravel()
        novelty = d_self / max(self.self_distance_scale_, 1e-12)
        if detectors.centers.size == 0:
            return self.novelty_score_weight * novelty

        d_det = _pairwise_euclidean(X, detectors.centers)
        margin = detectors.radii.reshape(1, -1) - d_det
        denom = np.maximum(detectors.radii.reshape(1, -1), 1e-12)
        detector_hit_score = np.maximum(margin / denom, 0.0).max(axis=1)

        if self._variant_ in {"densa", "antigen_nsa"}:
            # Density-focused variants emphasize sparse self neighborhoods.
            return detector_hit_score + 0.60 * novelty
        if self._variant_ in {"hnsa_idsa", "nsa_ii"}:
            return detector_hit_score + 0.45 * novelty
        return detector_hit_score + self.novelty_score_weight * novelty

    def _score_ensemble(self, X: np.ndarray) -> np.ndarray:
        scores = []
        for model in self.ensemble_detectors_:
            s = model.decision_function(X)
            s = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)
            scores.append(s)
        return np.mean(np.vstack(scores), axis=0)


class BinaryNSA(NegativeSelection):
    """Binary/string NSA with Hamming, r-contiguous-bit, or r-chunk matching."""

    def __init__(self, contamination=0.1, n_detectors=128, n_bits=32, r=4, match_rule="hamming", random_state=None):
        super(BinaryNSA, self).__init__(contamination=contamination, variant="bnsa", n_detectors=n_detectors, n_bits=n_bits, r=r, match_rule=match_rule, random_state=random_state)


class RNSA(NegativeSelection):
    """Random real-valued NSA with fixed-radius detectors."""

    def __init__(self, contamination=0.1, n_detectors=128, radius=None, sampling_margin=0.5, random_state=None):
        super(RNSA, self).__init__(contamination=contamination, variant="rnsa", n_detectors=n_detectors, radius=radius, sampling_margin=sampling_margin, random_state=random_state)


class RRNSA(RNSA):
    """Monte-Carlo random real-valued NSA alias."""

    def __init__(self, contamination=0.1, n_detectors=128, radius=None, sampling_margin=0.5, random_state=None):
        super(RRNSA, self).__init__(contamination=contamination, n_detectors=n_detectors, radius=radius, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "rrnsa"


class VDetector(NegativeSelection):
    """Variable-radius real-valued NSA / V-detector."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(VDetector, self).__init__(contamination=contamination, variant="vdetector", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class GridNSA(NegativeSelection):
    """Grid-based NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, n_grid=8, sampling_margin=0.5, random_state=None):
        super(GridNSA, self).__init__(contamination=contamination, variant="gnsa", n_detectors=n_detectors, n_grid=n_grid, sampling_margin=sampling_margin, random_state=random_state)


class GFRNSA(GridNSA):
    """Grid-file real-valued NSA alias."""

    def __init__(self, contamination=0.1, n_detectors=128, n_grid=8, sampling_margin=0.5, random_state=None):
        super(GFRNSA, self).__init__(contamination=contamination, n_detectors=n_detectors, n_grid=n_grid, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "gf_rnsa"


class MatrixNSA(GridNSA):
    """Matrix-representation NSA alias based on grid cells."""

    def __init__(self, contamination=0.1, n_detectors=128, n_grid=8, sampling_margin=0.5, random_state=None):
        super(MatrixNSA, self).__init__(contamination=contamination, n_detectors=n_detectors, n_grid=n_grid, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "matrix_nsa"


class ANSA(NegativeSelection):
    """Adaptive/self-adaptive NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(ANSA, self).__init__(contamination=contamination, variant="ansa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class EvoSeedRNSA(NegativeSelection):
    """Evolutionary seed real-valued NSA approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(EvoSeedRNSA, self).__init__(contamination=contamination, variant="evoseedrnsa", n_detectors=n_detectors, optimization_iter=optimization_iter, sampling_margin=sampling_margin, random_state=random_state)


class ORNSA(NegativeSelection):
    """Outlier-robust NSA using an internal/boundary self subset."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(ORNSA, self).__init__(contamination=contamination, variant="ornsa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class OptimizedNSA(NegativeSelection):
    """Optimized NSA with lightweight detector-position improvement."""

    def __init__(self, contamination=0.1, n_detectors=128, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(OptimizedNSA, self).__init__(contamination=contamination, variant="optimized_nsa", n_detectors=n_detectors, optimization_iter=optimization_iter, sampling_margin=sampling_margin, random_state=random_state)


class FtNSA(NegativeSelection):
    """Feature/self-space reduction NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(FtNSA, self).__init__(contamination=contamination, variant="ftnsa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class IVRNSA(NegativeSelection):
    """Improved variable-radius NSA with mature-detector pruning."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(IVRNSA, self).__init__(contamination=contamination, variant="ivrnsa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class CBNSA(NegativeSelection):
    """Clustering-based NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, n_clusters=12, sampling_margin=0.5, random_state=None):
        super(CBNSA, self).__init__(contamination=contamination, variant="cb_nsa", n_detectors=n_detectors, n_clusters=n_clusters, sampling_margin=sampling_margin, random_state=random_state)


class PRR2NSA(CBNSA):
    """Pattern-recognition-receptor dual NSA approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, n_clusters=12, sampling_margin=0.5, random_state=None):
        super(PRR2NSA, self).__init__(contamination=contamination, n_detectors=n_detectors, n_clusters=n_clusters, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "prr2nsa"


class DENSElectionNSA(NegativeSelection):
    """Density-enhanced NSA / DENSA."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(DENSElectionNSA, self).__init__(contamination=contamination, variant="densa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class DENSA(DENSElectionNSA):
    """Statistical-confidence / density detector placement NSA alias."""


class AntigenNSA(DENSElectionNSA):
    """Antigen-density NSA alias."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(AntigenNSA, self).__init__(contamination=contamination, n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "antigen_nsa"


class NSADE(NegativeSelection):
    """Differential-evolution NSA approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(NSADE, self).__init__(contamination=contamination, variant="nsa_de", n_detectors=n_detectors, optimization_iter=optimization_iter, sampling_margin=sampling_margin, random_state=random_state)


class NSAPSO(NegativeSelection):
    """Particle-swarm-optimization NSA approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(NSAPSO, self).__init__(contamination=contamination, variant="nsa_pso", n_detectors=n_detectors, optimization_iter=optimization_iter, sampling_margin=sampling_margin, random_state=random_state)


class IORNSA(NegativeSelection):
    """Immune-optimization real-valued NSA approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(IORNSA, self).__init__(contamination=contamination, variant="io_rnsa", n_detectors=n_detectors, optimization_iter=optimization_iter, sampling_margin=sampling_margin, random_state=random_state)


class BIORVNSA(NegativeSelection):
    """Bidirectional-inhibition real-valued NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(BIORVNSA, self).__init__(contamination=contamination, variant="biorv_nsa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class HNSAIDSA(NegativeSelection):
    """Hybrid NSA for adaptive intrusion-detection-style semi-supervised use."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(HNSAIDSA, self).__init__(contamination=contamination, variant="hnsa_idsa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class NSAII(HNSAIDSA):
    """NSA-II alias for hybrid self/non-self detector training."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(NSAII, self).__init__(contamination=contamination, n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "nsa_ii"


class OALFBNSA(NegativeSelection):
    """Online adaptive learning feedback NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(OALFBNSA, self).__init__(contamination=contamination, variant="oalfb_nsa", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class FBNSA(OALFBNSA):
    """Feedback NSA alias."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(FBNSA, self).__init__(contamination=contamination, n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "fb_nsa"


class MNSA(NegativeSelection):
    """Multiple NSA ensemble."""

    def __init__(self, contamination=0.1, n_detectors=128, n_estimators=5, sampling_margin=0.5, random_state=None):
        super(MNSA, self).__init__(contamination=contamination, variant="mnsa", n_detectors=n_detectors, n_estimators=n_estimators, sampling_margin=sampling_margin, random_state=random_state)


class NSNAD(NegativeSelection):
    """Feature-subspace negative-selection anomaly detector."""

    def __init__(self, contamination=0.1, n_detectors=128, feature_subsample=0.70, sampling_margin=0.5, random_state=None):
        super(NSNAD, self).__init__(contamination=contamination, variant="nsnad", n_detectors=n_detectors, feature_subsample=feature_subsample, sampling_margin=sampling_margin, random_state=random_state)


class RENNSA(NegativeSelection):
    """Reduced-overlap NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, sampling_margin=0.5, random_state=None):
        super(RENNSA, self).__init__(contamination=contamination, variant="ren", n_detectors=n_detectors, sampling_margin=sampling_margin, random_state=random_state)


class AINSA(NegativeSelection):
    """Adaptive-immunology NSA approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(AINSA, self).__init__(contamination=contamination, variant="ainsa", n_detectors=n_detectors, optimization_iter=optimization_iter, sampling_margin=sampling_margin, random_state=random_state)


class ODNSA(NegativeSelection):
    """Optimized detector-radius NSA."""

    def __init__(self, contamination=0.1, n_detectors=128, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(ODNSA, self).__init__(contamination=contamination, variant="odnsa", n_detectors=n_detectors, optimization_iter=optimization_iter, sampling_margin=sampling_margin, random_state=random_state)


class CNSA(CBNSA):
    """Clustered NSA / clustered fruit-fly-style optimization approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, n_clusters=12, optimization_iter=15, sampling_margin=0.5, random_state=None):
        super(CNSA, self).__init__(contamination=contamination, n_detectors=n_detectors, n_clusters=n_clusters, sampling_margin=sampling_margin, random_state=random_state)
        self.variant = "cnsa"
        self.optimization_iter = optimization_iter


class VORNSA(NegativeSelection):
    """Vectorized / big-data-oriented NSA approximation."""

    def __init__(self, contamination=0.1, n_detectors=128, radius=None, sampling_margin=0.5, random_state=None):
        super(VORNSA, self).__init__(contamination=contamination, variant="vor_nsa", n_detectors=n_detectors, radius=radius, sampling_margin=sampling_margin, random_state=random_state)


__all__ = [
    "NegativeSelection",
    "BinaryNSA",
    "RNSA",
    "RRNSA",
    "VDetector",
    "GridNSA",
    "GFRNSA",
    "MatrixNSA",
    "ANSA",
    "EvoSeedRNSA",
    "ORNSA",
    "OptimizedNSA",
    "FtNSA",
    "IVRNSA",
    "CBNSA",
    "PRR2NSA",
    "DENSElectionNSA",
    "DENSA",
    "AntigenNSA",
    "NSADE",
    "NSAPSO",
    "IORNSA",
    "BIORVNSA",
    "HNSAIDSA",
    "NSAII",
    "OALFBNSA",
    "FBNSA",
    "MNSA",
    "NSNAD",
    "RENNSA",
    "AINSA",
    "ODNSA",
    "CNSA",
    "VORNSA",
]
