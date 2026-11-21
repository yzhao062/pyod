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
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

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

    novelty : bool (default=True)
        Forwarded to scikit-learn's ``LocalOutlierFactor``. PyOD defaults to
        ``True`` (the opposite of scikit-learn's default of ``False``)
        because the PyOD ``BaseDetector`` contract is fit-on-train then
        ``predict``/``decision_function`` on unseen data, which scikit-learn
        only permits in novelty mode. Set ``novelty=False`` only if you want
        to read training-time scores from
        ``self.detector_.negative_outlier_factor_`` and never call
        ``predict`` or ``decision_function`` on new data — see scikit-learn's
        :class:`sklearn.neighbors.LocalOutlierFactor` for details.

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

    labels_ : numpy array of shape (n_samples,)
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

        # Clear explainability caches so a refit never reuses neighbors
        # or scores computed on a previous training set
        self._cached_1d_lof_scores = {}  # {dimension: lof_scores_array}
        if hasattr(self, 'knn_neighbors_'):
            del self.knn_neighbors_

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

        X = check_array(X, accept_sparse=True)
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
        """Lazily compute and cache the kNN of every training sample.

        ``kneighbors()`` is called without ``X`` so that, exactly as in
        scikit-learn's ``LocalOutlierFactor.fit``, a sample is not returned
        as its own neighbor. The neighbors are therefore identical to the
        ones behind ``decision_scores_`` and are shared across all
        dimensional explanations. The cache is cleared by ``fit``.
        """
        check_is_fitted(self, ['detector_'])
        if not hasattr(self, 'knn_neighbors_'):
            _, knn_inds = self.detector_.kneighbors()
            self.knn_neighbors_ = knn_inds  # shape: (n_samples, k)

    def _get_column(self, dimension):
        """Return one training feature as a dense 1D float array."""
        col = self.X_train_[:, [dimension]]
        if hasattr(col, 'toarray'):  # scipy sparse input
            col = col.toarray()
        return np.asarray(col, dtype=float).ravel()

    def _compute_lof_1d(self, dimension):
        """1D LOF score of every training sample along one dimension.

        Uses the overall (full-space) k nearest neighbors and computes
        k-distance, reachability distance, local reachability density and
        LOF along ``dimension`` only, with the same formulas as
        scikit-learn. For one-dimensional data the result equals the
        overall LOF score.

        Parameters
        ----------
        dimension : int
            Feature index.

        Returns
        -------
        lof_1d : numpy array of shape (n_samples,)
        """
        if dimension not in self._cached_1d_lof_scores:
            self._ensure_overall_neighbors()
            neighbors = self.knn_neighbors_  # (n_samples, k)
            x = self._get_column(dimension)  # (n_samples,)

            dists = np.abs(x[neighbors] - x[:, np.newaxis])  # (n_samples, k)
            k_dist = dists.max(axis=1)  # k-distance in this dimension
            reach_dists = np.maximum(dists, k_dist[neighbors])
            # 1e-10 avoids division by zero for duplicated values, as in
            # sklearn.neighbors.LocalOutlierFactor
            lrd = 1.0 / (np.mean(reach_dists, axis=1) + 1e-10)
            self._cached_1d_lof_scores[dimension] = (
                np.mean(lrd[neighbors], axis=1) / lrd)
        return self._cached_1d_lof_scores[dimension]

    def _check_explain_args(self, ind, columns):
        check_is_fitted(self, ['X_train_', 'detector_', 'decision_scores_'])
        n_samples, n_features = self.X_train_.shape
        if not -n_samples <= ind < n_samples:
            raise IndexError(
                "Sample index {} out of range [0, {})".format(ind, n_samples))
        ind = int(ind) % n_samples
        if columns is None:
            columns = list(range(n_features))
        else:
            columns = list(columns)
        return ind, columns

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
        ind, columns = self._check_explain_args(ind, columns)
        return np.array([self._compute_lof_1d(dim)[ind] for dim in columns])

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
        if not cutoff_bands:
            return '#1f77b4'

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
            if not chunk_cutoff_bands:
                break
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
        if by_label:
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

        ind, columns = self._check_explain_args(ind, columns)
        n_features = len(columns)

        # Set default cutoffs
        if cutoffs is None:
            cutoffs = [1 - self.contamination, 0.99]
        cutoffs = sorted(cutoffs)

        dim_scores = self.get_outlier_explainability_scores(ind, columns)

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

            # Dimensional LOF scores of ALL samples (cached per dimension)
            all_dim_scores = np.column_stack(
                [self._compute_lof_1d(dim) for dim in columns])

            # Cutoff bands: quantiles across all samples per dimension
            cutoff_bands = {
                cutoff: np.quantile(all_dim_scores, q=cutoff, axis=0)
                for cutoff in cutoffs}
        else:
            cutoff_bands = None

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
                chunk_cutoff_bands = None
                if cutoff_bands:
                    chunk_cutoff_bands = {
                        cutoff: cutoff_bands[cutoff][start_idx:end_idx]
                        for cutoff in cutoffs}
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
