# -*- coding: utf-8 -*-
"""Fast Local Outlier Factor (FastLOF) for Outlier Detection
"""
# Author: Alaa Abdelwahab
# License: BSD 2 clause

import math
import numpy as np
from numba import njit, prange
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances

from .base import BaseDetector


@njit
def _sift_down(dists, neighbors, start, end):
    """Max-heap sift down operation."""
    root = start
    while True:
        child = 2 * root + 1
        if child >= end:
            break

        # Find larger child
        if child + 1 < end and dists[child] < dists[child + 1]:
            child += 1

        # Check if we need to swap
        if dists[root] < dists[child]:
            dists[root], dists[child] = dists[child], dists[root]
            neighbors[root], neighbors[child] = neighbors[child], neighbors[root]
            root = child
        else:
            break


@njit(parallel=True)
def _update_neighbors(neighbors, neighbor_dists, qpos_active, tpos, M, k):
    """
    Use max-heap to track k-smallest. O(n_target * log k) per query.
    Best when k << n_target (e.g., k=10, n_target=1000).
    """
    n_active = qpos_active.shape[0]
    n_target = tpos.shape[0]

    for i in prange(n_active):
        q_idx = qpos_active[i]

        # Build max-heap from current k neighbors
        heap_dists = neighbor_dists[q_idx].copy()
        heap_neighbors = neighbors[q_idx].copy()

        # Process new candidates
        for j in range(n_target):
            if M[i, j] < heap_dists[0]:  # Better than worst in heap
                heap_dists[0] = M[i, j]
                heap_neighbors[0] = tpos[j]
                _sift_down(heap_dists, heap_neighbors, 0, k)

        neighbor_dists[q_idx] = heap_dists
        neighbors[q_idx] = heap_neighbors


@njit(parallel=True, cache=True, fastmath=True)
def _compute_distances_numba(X, Y, metric_type, p, is_symmetric):
    """
    Unified distance computation function.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features)
    metric_type : int
        0 = squared euclidean, 1 = manhattan, 2 = minkowski
    p : float
        Minkowski parameter (only used when metric_type=2)
    is_symmetric : bool
        If True, X and Y are the same and only upper triangle is computed
    """
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    n_features = X.shape[1]

    distances = np.empty((n_X, n_Y), dtype=X.dtype)

    if is_symmetric:
        # Symmetric case: compute upper triangle only
        for i in prange(n_X):
            distances[i, i] = 0.0

            for j in range(i + 1, n_X):
                dist = 0.0

                if metric_type == 0:  # squared euclidean
                    for k in range(n_features):
                        diff = X[i, k] - X[j, k]
                        dist += diff * diff
                elif metric_type == 1:  # manhattan
                    for k in range(n_features):
                        dist += abs(X[i, k] - X[j, k])
                else:  # minkowski
                    for k in range(n_features):
                        dist += abs(X[i, k] - X[j, k]) ** p
                    dist = dist ** (1.0 / p)

                distances[i, j] = dist
                distances[j, i] = dist
    else:
        # Non-symmetric case: compute full matrix
        for i in prange(n_X):
            for j in range(n_Y):
                dist = 0.0

                if metric_type == 0:  # squared euclidean
                    for k in range(n_features):
                        diff = X[i, k] - Y[j, k]
                        dist += diff * diff
                elif metric_type == 1:  # manhattan
                    for k in range(n_features):
                        dist += abs(X[i, k] - Y[j, k])
                else:  # minkowski
                    for k in range(n_features):
                        dist += abs(X[i, k] - Y[j, k]) ** p
                    dist = dist ** (1.0 / p)

                distances[i, j] = dist

    return distances


class FastLOF(BaseDetector):
    """Fast Local Outlier Factor (FastLOF) for outlier detection.

    FastLOF uses a chunked, iterative approach to compute Local Outlier Factor
    scores more efficiently than standard LOF for large datasets. It divides
    the dataset into chunks and computes nearest neighbors incrementally,
    with optional threshold-based filtering to focus computation on likely outliers.
    See :cite:`breunig2000lof,goldstein2012fastlof` for details.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use for LOF computation.

    algorithm : {'auto', 'brute'}, optional (default='auto')
        Algorithm to use for nearest neighbor search. Currently only 'brute'
        is fully implemented. 'auto' defaults to 'brute'.

    leaf_size : int, optional (default=30)
        Leaf size passed to tree-based neighbors. Currently not used
        (reserved for future enhancement).

    metric : str, optional (default='euclidean')
        Metric to use for distance computation. Supported metrics include:
        'euclidean', 'minkowski', 'manhattan', 'l1', 'l2', 'cosine', 'cityblock',
        and any metric supported by scikit-learn's pairwise_distances.

    p : int, optional (default=2)
        Parameter for Minkowski metric. When p=1, equivalent to Manhattan distance.
        When p=2, equivalent to Euclidean distance.

    metric_params : dict, optional (default=None)
        Additional keyword arguments for the metric function.
        Passed to scikit-learn's pairwise_distances for custom metrics.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e., the proportion
        of outliers in the data set. Used when fitting to define the
        threshold on the decision function.

    n_jobs : int, optional (default=None)
        Number of parallel jobs for computation. Currently validated but not
        implemented (reserved for future enhancement).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. Used for chunk shuffling to ensure reproducible results.

    chunk_size : int, optional (default=None)
        Size of data chunks for processing. If None, automatically calculated.

    threshold : float, optional (default=1.01)
        LOF threshold for considering points as potential outliers during
        iterative refinement. Points with LOF > threshold are processed
        in subsequent iterations. Default 1.01 means points slightly above
        normal density are considered.

    Attributes
    ----------
    n_neighbors_ : int
        The actual number of neighbors used for computation.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30,
                 metric='euclidean', p=2, metric_params=None,
                 contamination=0.1, n_jobs=None, random_state=None,
                 chunk_size=None, threshold=1.1):
        super(FastLOF, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit the FastLOF detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate inputs
        X = check_array(X)
        self._set_n_classes(y)

        n_samples = X.shape[0]

        # Validate and adjust n_neighbors (match scikit-learn behavior)
        if self.n_neighbors >= n_samples:
            self.n_neighbors_ = n_samples - 1
        else:
            self.n_neighbors_ = self.n_neighbors

        # Validate algorithm
        if self.algorithm not in ['auto', 'brute']:
            raise ValueError(f"algorithm must be 'auto' or 'brute', got {self.algorithm}")

        # Validate p parameter
        if self.p <= 0:
            raise ValueError(f"p must be positive, got {self.p}")

        # Validate leaf_size (for future tree support)
        if self.leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {self.leaf_size}")

        # Validate n_jobs
        if self.n_jobs is not None and self.n_jobs < -1:
            raise ValueError(f"n_jobs must be None, -1, or positive, got {self.n_jobs}")

        # Store training data for novelty detection
        self.X_train_ = X.copy()

        # Run FastLOF algorithm
        self.decision_scores_ = self._fastlof_compute(X)

        # Process decision scores to get threshold and labels
        self._process_decision_scores()

        return self

    def _calculate_chunk_size(self, n_samples, k):
        """Calculate chunk size to aim for log2(n_samples) chunks.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
        k : int
            Number of neighbors.

        Returns
        -------
        chunk_size : int
            Chunk size for processing, aiming for log2(n_samples) chunks.
        """
        if self.chunk_size is not None:
            return int(max(self.chunk_size, k + 1))

        # Always ensure chunk_size >= k + 1
        min_chunk_size = k + 1

        return int(max(min_chunk_size, min(pow(math.log10(n_samples), 5), 40000)))

    def _compute_distances(self, X, Y, is_symmetric=False):
        """
        Compute distances between X and Y.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First array of samples.

        Y : ndarray of shape (n_samples_Y, n_features)
            Second array of samples.

        is_symmetric : bool, default=False
            If True, assumes X and Y are the same array and only computes
            upper triangle for efficiency.

        Returns
        -------
        distances : ndarray of shape (n_samples_X, n_samples_Y)
            Distance matrix between X and Y.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)
        # Determine metric type for numba function
        if self.metric == 'euclidean' or (self.metric == 'minkowski' and self.p == 2):
            metric_type = 0  # squared euclidean
        elif self.metric == 'manhattan' or self.metric == 'cityblock' or (self.metric == 'minkowski' and self.p == 1):
            metric_type = 1  # manhattan
        elif self.metric == 'minkowski':
            metric_type = 2  # minkowski
        else:
            metric_type = -1  # fall back to scipy/sklearn

        # Use optimized numba function for supported metrics
        if metric_type >= 0:
            return _compute_distances_numba(X, Y, metric_type, self.p, is_symmetric)

        # Fall back to scipy/sklearn for unsupported metrics
        try:
            params = self.metric_params or {}
            if self.metric == 'minkowski':
                params['p'] = self.p
            return cdist(X, Y, metric=self.metric, **params)
        except (ValueError, ImportError, TypeError):
            params = self.metric_params or {}
            if self.metric == 'minkowski':
                params['p'] = self.p
            return pairwise_distances(X, Y, metric=self.metric, **params)

    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed as the Local Outlier
        Factor with respect to the training samples. For consistency, outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are not supported.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        if not hasattr(self, 'decision_scores_'):
            raise ValueError("FastLOF must be fitted before decision_function")

        # Compute LOF scores using the training data structure
        return self._compute_lof_scores(X)

    def _fastlof_compute(self, X):
        """
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        lof_scores : numpy array of shape (n_samples,)
            The LOF scores for all samples. Higher scores indicate outliers.
        """
        # ===== INITIALIZATION =====
        # Get number of samples and number of neighbors
        n_samples = X.shape[0]
        k = self.n_neighbors_
        epsilon = 1e-10  # For convergence check

        # Set chunk size dynamically
        chunk_size = self._calculate_chunk_size(n_samples, k)

        # Random state for reproducibility
        if self.random_state is None:
            rng = np.random.RandomState()
        else:
            # Handle both integer and RandomState objects
            if hasattr(self.random_state, 'get_state'):
                rng = self.random_state
            else:
                rng = np.random.RandomState(self.random_state)
        # Get a random permutation of the data
        perm = rng.permutation(n_samples)

        # Build chunks from the permuted data
        chunks = [perm[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]

        # Ensure the last chunk is at least k+1 in size by merging with previous if needed
        if len(chunks) > 1 and len(chunks[-1]) < k + 1:
            # Merge last chunk with previous chunk
            chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
            chunks.pop()

        # Get the number of chunks
        n_chunks = len(chunks)

        # Pre-compute chunk position arrays based on actual chunks
        chunk_positions = []
        start_idx = 0
        for chunk in chunks:
            chunk_len = len(chunk)
            chunk_positions.append(np.arange(start_idx, start_idx + chunk_len, dtype=np.int32))
            start_idx += chunk_len

        # Initialize neighbor storage to -1 for all points
        neighbors = np.full((n_samples, k), -1, dtype=np.int32)
        # Initialize neighbor distances to infinity for all points
        neighbor_dists = np.full((n_samples, k), np.inf, dtype=np.float32)

        # Initialize all points as active to 1
        active = np.ones(n_samples, dtype=bool)
        # Initialize LOF scores to 1 for all points
        lof = np.ones(n_samples, dtype=np.float32)

        # ===== MAIN CHUNK PROCESSING LOOP =====
        for offset in range(n_chunks):
            # Store old k-distances before this diagonal pass
            old_kdist = neighbor_dists[:, 0].copy()

            # Process all chunk pairs at this offset distance
            for i in range(n_chunks - offset):
                j = i + offset

                qpos = chunk_positions[i]
                tpos = chunk_positions[j]

                # Get active mask for chunk i
                active_mask = active[qpos]
                # If there are no more active points, stop
                if not np.any(active_mask) or len(tpos) == 0:
                    continue

                # Get active points in chunk i
                qpos_active = qpos[active_mask]

                # Get indices of active points in chunk i and all points in chunk j
                q_indices = chunks[i][active_mask]
                t_indices = chunks[j]
                M = self._compute_distances(X[q_indices], X[t_indices], is_symmetric=(i == j))

                # Handle self-distances
                if i == j:
                    # Set self-distances to infinity
                    np.fill_diagonal(M, np.inf)

                # Update neighbors for chunk i (active points only)
                # Use the neighbor indices, neighbor distances, active points in chunk i, and all points in chunk j,
                # and the distance matrix, and the number of neighbors to update the neighbors and neighbor distances
                _update_neighbors(neighbors, neighbor_dists, qpos_active, tpos, M, k)

                # Symmetric update for chunk j (if different chunks)
                if i != j:
                    _update_neighbors(neighbors, neighbor_dists, tpos, qpos_active, M.T, k)

            # Compute LOF scores after processing this diagonal
            lof, lrd = self._compute_lof(neighbors, neighbor_dists, k)

            # Update active set
            if offset >= 1:
                active = lof > self.threshold

            # Check for improvements (smart convergence)
            new_kdist = neighbor_dists[:, 0]
            improvements = np.sum(new_kdist < old_kdist - epsilon)

            # Early stopping if no improvements
            if improvements == 0:
                break

        # ===== FINALIZATION =====
        # Remap LOF scores back to original order
        lof_orig = np.empty_like(lof)
        lof_orig[perm] = lof
        lrd_orig = np.empty_like(lrd)
        lrd_orig[perm] = lrd

        self.lrd_ = lrd_orig
        return lof_orig

    def _compute_lof(self, neighbors, neighbor_dists, k):
        """Compute Local Outlier Factor scores.

        Parameters
        ----------
        neighbors : numpy array of shape (n_samples, k)
            Neighbor indices for each point.

        neighbor_dists : numpy array of shape (n_samples, k)
            Distances to neighbors (squared for euclidean/metric='minkowski' with p=2).

        k : int
            Number of neighbors.

        Returns
        -------
        lof : numpy array of shape (n_samples,)
            LOF scores for all points.
        """
        # Convert squared distances to actual distances for euclidean metrics
        if self.metric == 'euclidean' or (self.metric == 'minkowski' and self.p == 2):
            neighbor_dists = np.sqrt(neighbor_dists)

        # Get k-distances (distance to k-th nearest neighbor)
        kdist = neighbor_dists[:, 0].copy()

        # Compute reachability distances
        # reach_dist[i,j] = max(kdist[j], dist(i,j))
        neighbor_kdists = kdist[neighbors]
        reach_dists = np.maximum(neighbor_dists, neighbor_kdists)

        # Local Reachability Density (LRD)
        # Match scikit-learn's approach: add small epsilon to avoid division by zero
        mean_reach_dist = np.mean(reach_dists, axis=1)
        lrd = 1.0 / (mean_reach_dist + 1e-10)

        # LOF = mean(lrd[neighbors]) / lrd[point]
        neighbor_lrds = lrd[neighbors]
        mean_neighbor_lrd = np.mean(neighbor_lrds, axis=1)

        # LOF computation
        lof = mean_neighbor_lrd / lrd

        return lof, lrd

    def _compute_lof_scores(self, X):
        """Compute LOF scores for new data using the training data structure.

        This method implements novelty detection by computing LOF scores
        for new data points using the existing training data. It employs
        lazy initialization for the nearest neighbor searcher to ensure
        memory efficiency.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            New data points to score.

        Returns
        -------
        lof_scores : numpy array of shape (n_samples,)
            LOF scores for the new data points.
        """
        # Lazy initialization: Initialize the neighbor searcher only if needed
        if not hasattr(self, 'nbrs_'):
            self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors_,
                                          algorithm=self.algorithm,
                                          metric=self.metric,
                                          p=self.p,
                                          metric_params=self.metric_params,
                                          n_jobs=self.n_jobs)
            self.nbrs_.fit(self.X_train_)

        # Find k nearest neighbors in training data
        # dists: Distances to the k nearest neighbors
        # indices: Indices of the k nearest neighbors in X_train
        dists, indices = self.nbrs_.kneighbors(X)

        # Compute reachability distances for new points
        # rd_new_to_train = max(k_distance, distance)
        k_distances_new = dists[:, -1]  # k-distance of each new point

        rd_new_to_train = np.maximum(
            k_distances_new[:, np.newaxis],  # Broadcast k-distance to match distances shape
            dists
        )

        # Compute local reachability density (LRD) for new points
        # Add small epsilon to avoid division by zero
        mean_reach_dist_new = np.mean(rd_new_to_train, axis=1)
        lrd_new = 1.0 / (mean_reach_dist_new + 1e-10)

        # Retrieve pre-computed LRDs of the neighbors from training
        # This utilizes the LRDs stored during fit() to avoid re-computation
        neighbor_lrd = self.lrd_[indices]

        # Compute average LRD of neighbors
        mean_neighbor_lrd = np.mean(neighbor_lrd, axis=1)

        # Compute final LOF scores
        lof_scores = mean_neighbor_lrd / lrd_new

        return lof_scores
