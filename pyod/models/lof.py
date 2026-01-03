# -*- coding: utf-8 -*-
"""Local Outlier Factor (LOF). Implemented on scikit-learn library.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ..utils.utility import invert_order


# noinspection PyProtectedMember


class LOF(BaseDetector):
    """Wrapper of scikit-learn LOF Class with more functionalities.
    Unsupervised Outlier Detection using Local Outlier Factor (LOF).

    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered outliers.
    See :cite:`breunig2000lof` for details.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use by default for `kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default=30)
        Leaf size passed to `BallTree` or `KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If 'precomputed', the training input X is expected to be a distance
        matrix.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics:
        http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the decision function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.

    novelty : bool (default=False)
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        that you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set.

    Attributes
    ----------
    n_neighbors_ : int
        The actual number of neighbors used for `kneighbors` queries.

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
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1, novelty=True):
        super(LOF, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.novelty = novelty
        self._cached_1d_lof_scores = {} 
        self._cached_1d_k_distances = {} 

    # noinspection PyIncorrectDocstring
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

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
        # validate inputs X and y (optional)
        X = check_array(X, accept_sparse=True)
        self._set_n_classes(y)

        # Store training data for explainability
        self.X_train_ = X

        # Clear caches when fitting new data
        self._cached_1d_lof_scores = {}  # {dimension: lof_scores_array}
        self._cached_1d_k_distances = {}  # {dimension: k_distances_array}

        self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                            algorithm=self.algorithm,
                                            leaf_size=self.leaf_size,
                                            metric=self.metric,
                                            p=self.p,
                                            metric_params=self.metric_params,
                                            contamination=self.contamination,
                                            n_jobs=self.n_jobs,
                                            novelty=self.novelty)
        self.detector_.fit(X=X, y=y)

        # Invert decision_scores_. Outliers comes with higher outlier scores
        self.decision_scores_ = invert_order(
            self.detector_.negative_outlier_factor_)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        # Invert outlier scores. Outliers comes with higher outlier scores
        # noinspection PyProtectedMember
        try:
            return invert_order(self.detector_._score_samples(X))
        except AttributeError:
            try:
                return invert_order(self.detector_._decision_function(X))
            except AttributeError:
                return invert_order(self.detector_.score_samples(X))

    @property
    def n_neighbors_(self):
        """The actual number of neighbors used for kneighbors queries.
        Decorator for scikit-learn LOF attributes.
        """
        return self.detector_.n_neighbors_

    def _ensure_overall_neighbors(self):
        """Lazily compute and cache overall kNN neighbors used for explainability.

        This avoids doing the extra kneighbors computation during fit, and instead
        performs it on first use by explanation routines. The neighbors are shared
        across all dimensional explanations to ensure consistency with the overall
        LOF model.
        """
        check_is_fitted(self, ['detector_'])
        if not hasattr(self, 'knn_neighbors_'):
            _, knn_inds = self.detector_.kneighbors(self.X_train_)
            self.knn_neighbors_ = knn_inds  # shape: (n_samples, k)
            
    def _precompute_k_distances_1d(self, dimension):
        """Pre-compute k-distances for all points in a 1D subspace.
        
        For each point, computes the maximum distance to its k-nearest neighbors
        (from overall LOF) in the specified dimension. Computed once per dimension
        and cached for reuse across all sample explanations.
        
        Parameters
        ----------
        dimension : int
            Single dimension index
            
        Returns
        -------
        k_distances_1d : numpy array of shape (n_samples,)
            k-distance for each point in this dimension
        """
        # Ensure overall neighbors are available (lazy computation)
        self._ensure_overall_neighbors()

        subspace_data = self.X_train_[:, [dimension]]  # (n_samples, 1)
        n_samples = self.X_train_.shape[0]
        k_distances_1d = np.zeros(n_samples)
        
        # Vectorized computation: for each point, compute max distance to its k-neighbors
        for point_idx in range(n_samples):
            point = subspace_data[point_idx]  # (1,)
            neighbor_indices = self.knn_neighbors_[point_idx]  # (k,)
            neighbor_points = subspace_data[neighbor_indices]  # (k, 1)
            
            # Vectorized distance computation
            dists = np.linalg.norm(neighbor_points - point, axis=1)  # (k,)
            k_distances_1d[point_idx] = np.max(dists)
        
        return k_distances_1d

    def _compute_lof_subspace_with_neighbors(self, sample_idx, dimensions, 
                                             subspace_data, k_distances_1d):
        """Compute LOF score for a sample in a subspace using overall LOF neighbors.
        
        Uses the same neighbors as the overall LOF model, but computes density metrics
        in the subspace defined by the specified dimensions. k-distances must be 
        pre-computed and passed in for efficiency.
        
        Parameters
        ----------
        sample_idx : int
            Index of the sample
        dimensions : list of int
            List of dimension indices defining the subspace
        subspace_data : numpy array of shape (n_samples, len(dimensions))
            Subspace data points
        k_distances_1d : numpy array of shape (n_samples,)
            Pre-computed k-distances for all points in this subspace
        
        Returns
        -------
        lof_subspace : float
            LOF score for this sample in this subspace
        """
        k = self.n_neighbors_
        
        # Get OVERALL LOF neighbors
        neighbor_indices = self.knn_neighbors_[sample_idx]
        
        # Step 1: Vectorized distances from sample to neighbors in subspace
        sample_point = subspace_data[sample_idx]  # (d,)
        neighbor_points = subspace_data[neighbor_indices]  # (k, d)
        dists_sample_to_neighbors = np.linalg.norm(
            neighbor_points - sample_point, axis=1
        )  # (k,)
        
        # Step 2: k-distances for neighbors (lookup from pre-computed cache)
        k_dists_neighbors = k_distances_1d[neighbor_indices]  # (k,)
        
        # Step 3: Compute reachability distances
        reach_dists = np.maximum(k_dists_neighbors, dists_sample_to_neighbors)
        
        # Handle edge case: if all reachability distances are zero (identical points)
        if np.sum(reach_dists) == 0:
            return 1.0
        
        # Step 4: Compute LRD for the sample
        lrd_sample = k / np.sum(reach_dists)
        
        # Step 5: Vectorized LRD computation for neighbors
        lrd_neighbors = np.zeros(len(neighbor_indices))
        
        for i, neighbor_idx in enumerate(neighbor_indices):
            # Get neighbor's neighbors from overall LOF
            neighbor_neighbor_indices = self.knn_neighbors_[neighbor_idx]
            
            # Vectorized distances from neighbor to its neighbors in subspace
            neighbor_point = subspace_data[neighbor_idx]  # (d,)
            neighbor_neighbor_points = subspace_data[neighbor_neighbor_indices]  # (k, d)
            dists_neighbor_to_neighbors = np.linalg.norm(
                neighbor_neighbor_points - neighbor_point, axis=1
            )  # (k,)
            
            # k-distances for neighbor's neighbors (lookup from pre-computed cache)
            k_dists_for_neighbor_neighbors = k_distances_1d[neighbor_neighbor_indices]  # (k,)
            
            # Compute reachability distances for neighbor
            reach_dists_neighbor = np.maximum(
                k_dists_for_neighbor_neighbors, dists_neighbor_to_neighbors
            )
            
            # Handle edge case: if all reachability distances are zero
            if np.sum(reach_dists_neighbor) == 0:
                lrd_neighbors[i] = 1.0
            else:
                # Compute LRD for this neighbor
                lrd_neighbors[i] = len(neighbor_neighbor_indices) / np.sum(reach_dists_neighbor)
        
        # Step 6: Compute LOF
        if lrd_sample == 0:
            return 1.0
        
        lof_subspace = np.mean(lrd_neighbors) / lrd_sample
        
        return lof_subspace

    def get_outlier_explainability_scores(self, ind, columns=None):
        """Compute per-feature LOF-based explainability scores for a training sample.
        
        For each requested dimension, computes a 1D LOF score using the same
        k-nearest neighbors as the overall LOF model. This mirrors the API of
        ``KNN.get_outlier_explainability_scores`` and can be used independently
        of plotting.
        
        Parameters
        ----------
        ind : int
            Index of the sample in the training data.
        columns : list, optional
            Specific feature indices. If None, use all features.
        
        Returns
        -------
        scores : numpy array of shape (n_selected_features,)
            1D LOF score per selected dimension.
        """
        check_is_fitted(self, ['X_train_', 'detector_'])
        self._ensure_overall_neighbors()

        # Validate ind parameter
        if not 0 <= ind < self.X_train_.shape[0]:
            raise IndexError(
                f"Sample index {ind} out of range [0, {self.X_train_.shape[0]})"
            )

        if columns is None:
            columns = list(range(self.X_train_.shape[1]))

        scores = np.zeros(len(columns))

        for idx, dim in enumerate(columns):
            # Ensure k-distances cache for this dimension
            if dim not in self._cached_1d_k_distances:
                self._cached_1d_k_distances[dim] = self._precompute_k_distances_1d(dim)
            k_distances_1d = self._cached_1d_k_distances[dim]

            # Subspace is just this dimension
            subspace_data = self.X_train_[:, [dim]]

            scores[idx] = self._compute_lof_subspace_with_neighbors(
                ind, [dim], subspace_data, k_distances_1d
            )

        return scores

    def _get_bar_color(self, score, cutoff_bands, cutoffs, idx):
        """Determine bar color based on score and cutoff thresholds.
        
        Parameters
        ----------
        score : float
            The dimensional LOF score
        cutoff_bands : dict
            Dictionary mapping cutoff values to arrays of thresholds
        cutoffs : list
            List of cutoff percentile values
        idx : int
            Index of the feature
            
        Returns
        -------
        color : str
            Hex color code for the bar
        """
        cutoff_99 = cutoff_bands[cutoffs[-1]][idx]
        
        if len(cutoffs) == 1:
            return '#d62728' if score > cutoff_99 else '#1f77b4'
        
        cutoff_90 = cutoff_bands[cutoffs[0]][idx]
        
        if cutoff_90 == cutoff_99:
            return '#d62728' if score > cutoff_99 else '#1f77b4'
        
        if score > cutoff_99:
            return '#d62728'  # Red - extreme
        elif score > cutoff_90:
            return '#ff7f0e'  # Orange - warning
        else:
            return '#1f77b4'  # Blue - normal

    def _plot_feature_chunk(self, ind, chunk_columns, chunk_dim_scores, 
                            chunk_cutoff_bands, chunk_feature_names, cutoffs,
                            chunk_idx, total_chunks, file_name=None, file_type=None):
        """Helper method to plot a single chunk of features for LOF.
        
        Parameters
        ----------
        ind : int
            Index of the sample.
        chunk_columns : list
            Column indices for this chunk.
        chunk_dim_scores : numpy array
            Dimensional scores for this chunk.
        chunk_cutoff_bands : dict
            Cutoff bands for this chunk.
        chunk_feature_names : list
            Feature names for this chunk.
        cutoffs : list
            Cutoff values.
        chunk_idx : int
            Current chunk index (0-based).
        total_chunks : int
            Total number of chunks.
        file_name : str, optional
            Base file name for saving.
        file_type : str, optional
            File type extension.
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        n_features_chunk = len(chunk_columns)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, n_features_chunk * 0.4)))
        
        y_pos = np.arange(n_features_chunk)
        
        # Determine bar colors based on cutoffs using helper method
        colors = [
            self._get_bar_color(score, chunk_cutoff_bands, cutoffs, idx)
            for idx, score in enumerate(chunk_dim_scores)
        ]

        # Plot horizontal bars
        bars = ax.barh(y_pos, chunk_dim_scores, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=0.5)

        # Add cutoff lines
        line_styles = ['--', '-.', ':']
        line_colors = ['#ff7f0e', '#d62728', '#8c564b']
        for idx, (cutoff, style, color) in enumerate(zip(cutoffs, 
                                                          line_styles[:len(cutoffs)],
                                                          line_colors[:len(cutoffs)])):
            cutoff_values = chunk_cutoff_bands[cutoff]
            # Plot individual cutoff values for each feature
            for i, val in enumerate(cutoff_values):
                ax.plot([val, val], [i - 0.4, i + 0.4], 
                       linestyle=style, color=color, linewidth=2,
                       label=f'{cutoff:.2f} Cutoff' if i == 0 else "")

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(chunk_feature_names)
        ax.set_xlabel('Dimensional LOF Score (1D Approximation)', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        # Determine label
        label = 'Outlier' if self.labels_[ind] == 1 else 'Inlier'
        
        # Get overall LOF score
        overall_lof_score = self.decision_scores_[ind]
        
        # Title with chunk information if multiple chunks
        if total_chunks > 1:
            title = (f'LOF Outlier Explanation for Sample #{ind + 1} ({label})\n'
                    f'k={self.n_neighbors} | Overall LOF={overall_lof_score:.3f} | '
                    f'Features {chunk_columns[0]+1}-{chunk_columns[-1]+1} '
                    f'(Part {chunk_idx+1}/{total_chunks})')
        else:
            title = (f'LOF Outlier Explanation for Sample #{ind + 1} ({label})\n'
                    f'k={self.n_neighbors} | Overall LOF={overall_lof_score:.3f} | '
                    f'Shows per-dimension density deviation')
        
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, chunk_dim_scores)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {score:.3f}',
                   ha='left', va='center', fontsize=9)

        # Remove duplicate legend entries and add it
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='best', framealpha=0.9)

        ax.grid(axis='x', alpha=0.3, linestyle=':')
        plt.tight_layout()

        # Save the file if specified
        if file_name is not None:
            if total_chunks > 1:
                # Add chunk suffix to filename
                save_name = f'{file_name}_part{chunk_idx+1}of{total_chunks}'
            else:
                save_name = file_name
                
            if file_type is not None:
                plt.savefig(f'{save_name}.{file_type}', dpi=300, 
                           bbox_inches='tight')
            else:
                plt.savefig(f'{save_name}.png', dpi=300, 
                           bbox_inches='tight')
        
        plt.show()

        return fig, ax

    def explain_outlier(self, ind, columns=None, cutoffs=None,
                        feature_names=None, file_name=None,
                        file_type=None, max_features_per_plot=20,
                        compute_cutoffs=True):  # pragma: no cover
        """Plot dimensional outlier graph for a given data point within
        the dataset. Shows per-feature LOF scores computed using neighbors from overall LOF.

        For LOF, this method computes a 1D LOF score for each dimension using the same
        k-nearest neighbors from the overall LOF model. This ensures consistency and
        accurately identifies which dimensions contribute to the sample's outlier score.

        For datasets with many features (>max_features_per_plot), the plot is
        automatically split into multiple subplots for better readability.

        Parameters
        ----------
        ind : int
            The index of the data point one wishes to obtain
            a dimensional outlier graph for.

        columns : list, optional (default=None)
            Specify a list of features/dimensions for plotting. If not 
            specified, use all features.
        
        cutoffs : list of floats in (0., 1), optional (default=[1-contamination, 0.99])
            The significance cutoff bands of the dimensional outlier graph.
            These represent percentile thresholds across all samples.
            If None, defaults to [1 - contamination, 0.99].
        
        feature_names : list of strings, optional (default=None)
            The display names of all columns of the dataset,
            to show on the y-axis of the plot.

        file_name : string, optional (default=None)
            The name to save the figure. For multi-chunk plots, suffixes
            will be added automatically (e.g., '_part1of3').

        file_type : string, optional (default=None)
            The file type to save the figure (e.g., 'png', 'pdf', 'svg').

        max_features_per_plot : int, optional (default=20)
            Maximum number of features to display in a single plot.
            If the number of features exceeds this, multiple plots will be created.

        compute_cutoffs : bool, optional (default=True)
            If True, computes LOF scores for all samples to generate cutoff bands.
            If False, only computes LOF score for the target sample (much faster).
            When True, results are cached for subsequent calls.

        Returns
        -------
        fig, ax : matplotlib figure and axes (or list of figures/axes for multi-chunk)
            The dimensional outlier graph(s) for data point with index ind.
            Returns the last figure/axes pair, or a list if multiple chunks.
            
        Notes
        -----
        The dimensional LOF scores use the same k-nearest neighbors from the overall
        LOF model, ensuring mathematical consistency. Density metrics (k-distance,
        reachability distance, LRD, LOF) are computed in each 1D subspace using those
        neighbors. This provides accurate insight into which dimensions exhibit unusual
        local density patterns that contribute to the overall outlier score.
        """
        check_is_fitted(self, ['X_train_', 'detector_', 'decision_scores_',
                               'threshold_', 'labels_'])

        # Validate ind parameter
        if not 0 <= ind < self.X_train_.shape[0]:
            raise IndexError(
                f"Sample index {ind} out of range [0, {self.X_train_.shape[0]})"
            )

        # Ensure neighbors are available for dimensional LOF computations
        self._ensure_overall_neighbors()

        # Determine which columns to use
        if columns is None:
            columns = list(range(self.X_train_.shape[1]))
            n_features = self.X_train_.shape[1]
        else:
            n_features = len(columns)

        # Set default cutoffs
        if cutoffs is None:
            cutoffs = [1 - self.contamination, 0.99]

        if compute_cutoffs:
            # Check memory usage and warn if large
            n_samples = self.X_train_.shape[0]
            estimated_memory_mb = (n_samples * n_features * 8) / (1024 * 1024)
            if estimated_memory_mb > 1000:
                warnings.warn(
                    f"Computing cutoff bands will use ~{estimated_memory_mb:.0f}MB of memory. "
                    f"Consider using compute_cutoffs=False for faster computation.",
                    ResourceWarning
                )

            # Compute dimensional LOF scores for ALL samples (for cutoff bands)
            all_dim_scores = np.zeros((self.X_train_.shape[0], n_features))
            
            for dim_idx, dim in enumerate(columns):
                # Check cache first
                if dim in self._cached_1d_lof_scores:
                    all_dim_scores[:, dim_idx] = self._cached_1d_lof_scores[dim]
                else:
                    # Pre-compute k-distances for this dimension (once, then reuse)
                    if dim not in self._cached_1d_k_distances:
                        self._cached_1d_k_distances[dim] = self._precompute_k_distances_1d(dim)
                    k_distances_1d = self._cached_1d_k_distances[dim]
                    
                    # Extract subspace data once per dimension
                    subspace_data = self.X_train_[:, [dim]]  # shape: (n_samples, 1)
                    
                    # Compute for all samples (reusing pre-computed k-distances)
                    dim_scores_all = np.zeros(self.X_train_.shape[0])
                    for sample_idx in range(self.X_train_.shape[0]):
                        dim_scores_all[sample_idx] = self._compute_lof_subspace_with_neighbors(
                            sample_idx, [dim], subspace_data, k_distances_1d
                        )
                    
                    all_dim_scores[:, dim_idx] = dim_scores_all
                    
                    # Cache the results
                    self._cached_1d_lof_scores[dim] = dim_scores_all.copy()
            
            # Extract scores for the target sample
            dim_scores = all_dim_scores[ind, :]
            
            # Compute cutoff bands (quantiles across all samples per dimension)
            cutoff_bands = {}
            for cutoff in cutoffs:
                cutoff_bands[cutoff] = np.quantile(all_dim_scores, q=cutoff, axis=0)
        else:
            # Only compute for target sample (much faster, no cutoff bands)
            dim_scores = np.zeros(n_features)
            
            for dim_idx, dim in enumerate(columns):
                # Pre-compute k-distances for this dimension (once, then reuse)
                if dim not in self._cached_1d_k_distances:
                    self._cached_1d_k_distances[dim] = self._precompute_k_distances_1d(dim)
                k_distances_1d = self._cached_1d_k_distances[dim]
                
                # Extract subspace data for this dimension
                subspace_data = self.X_train_[:, [dim]]  # shape: (n_samples, 1)
                
                dim_scores[dim_idx] = self._compute_lof_subspace_with_neighbors(
                    ind, [dim], subspace_data, k_distances_1d
                )
            
            # No cutoff bands when compute_cutoffs=False
            cutoff_bands = {}
            for cutoff in cutoffs:
                cutoff_bands[cutoff] = np.zeros(n_features)

        # Set up feature names with validation
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in columns]
        else:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) does not match "
                    f"number of dimensions ({n_features})."
                )

        # Split into chunks if needed
        if n_features <= max_features_per_plot:
            # Single plot - use original logic
            chunk_columns_list = [columns]
            chunk_dim_scores_list = [dim_scores]
            chunk_cutoff_bands_list = [cutoff_bands]
            chunk_feature_names_list = [feature_names]
        else:
            # Split into chunks
            chunk_columns_list = []
            chunk_dim_scores_list = []
            chunk_cutoff_bands_list = []
            chunk_feature_names_list = []
            
            for start_idx in range(0, n_features, max_features_per_plot):
                end_idx = min(start_idx + max_features_per_plot, n_features)
                chunk_columns = columns[start_idx:end_idx]
                chunk_columns_list.append(chunk_columns)
                chunk_dim_scores_list.append(dim_scores[start_idx:end_idx])
                
                # Extract cutoff bands for this chunk
                chunk_cutoff_bands = {}
                for cutoff in cutoffs:
                    chunk_cutoff_bands[cutoff] = cutoff_bands[cutoff][start_idx:end_idx]
                chunk_cutoff_bands_list.append(chunk_cutoff_bands)
                
                chunk_feature_names_list.append(feature_names[start_idx:end_idx])

        # Create plots for each chunk
        total_chunks = len(chunk_columns_list)
        figures = []
        
        for chunk_idx, (chunk_columns, chunk_dim_scores, chunk_cutoff_bands, 
                       chunk_feature_names) in enumerate(zip(
            chunk_columns_list, chunk_dim_scores_list, 
            chunk_cutoff_bands_list, chunk_feature_names_list)):
            
            fig, ax = self._plot_feature_chunk(
                ind, chunk_columns, chunk_dim_scores, chunk_cutoff_bands,
                chunk_feature_names, cutoffs, chunk_idx, total_chunks,
                file_name, file_type
            )
            figures.append((fig, ax))

        # Return last figure/axes for backward compatibility, or all if multiple chunks
        if total_chunks == 1:
            return figures[0]
        else:
            return figures