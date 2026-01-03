# -*- coding: utf-8 -*-
"""k-Nearest Neighbors Detector (kNN)
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause


from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


# TODO: algorithm parameter is deprecated and will be removed in 0.7.6.
# Warning has been turned on.
# TODO: since Ball_tree is used by default, may introduce its parameters.

class KNN(BaseDetector):
    # noinspection PyPep8
    """kNN class for outlier detection.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
    details.

    Three kNN detectors are supported:
    largest: use the distance to the kth neighbor as the outlier score
    mean: use the average of all k neighbors as the outlier score
    median: use the median of the distance to k neighbors as the outlier score

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for k neighbors queries.

    method : str, optional (default='largest')
        {'largest', 'mean', 'median'}

        - 'largest': use the distance to the kth neighbor as the outlier score
        - 'mean': use the average of all k neighbors as the outlier score
        - 'median': use the median of the distance to k neighbors as the
          outlier score

    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for `radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

        .. deprecated:: 0.74
           ``algorithm`` is deprecated in PyOD 0.7.4 and will not be
           possible in 0.7.6. It has to use BallTree for consistency.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.

    Attributes
    ----------
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

    def __init__(self, contamination=0.1, n_neighbors=5, method='largest',
                 radius=1.0, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None, n_jobs=1,
                 **kwargs):
        super(KNN, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.method = method
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        if self.algorithm != 'auto' and self.algorithm != 'ball_tree':
            warn('algorithm parameter is deprecated and will be removed '
                 'in version 0.7.6. By default, ball_tree will be used.',
                 FutureWarning)

        self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                       radius=self.radius,
                                       algorithm=self.algorithm,
                                       leaf_size=self.leaf_size,
                                       metric=self.metric,
                                       p=self.p,
                                       metric_params=self.metric_params,
                                       n_jobs=self.n_jobs,
                                       **kwargs)
        # Cache for dimensional scores to avoid recomputation
        self._cached_dimensional_scores = {}  # {columns_tuple: scores_array}

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
        X = check_array(X)
        self._set_n_classes(y)

        # Store training data for explainability
        self.X_train_ = X

        # Clear cache when fitting new data
        self._cached_dimensional_scores = {}

        self.neigh_.fit(X)

        # In certain cases, _tree does not exist for NearestNeighbors
        # See Issue #158 (https://github.com/yzhao062/pyod/issues/158)
        # n_neighbors = 100
        if self.neigh_._tree is not None:
            self.tree_ = self.neigh_._tree

        else:
            if self.metric_params is not None:
                self.tree_ = BallTree(X, leaf_size=self.leaf_size,
                                      metric=self.metric,
                                      **self.metric_params)
            else:
                self.tree_ = BallTree(X, leaf_size=self.leaf_size,
                                      metric=self.metric)

        dist_arr, _ = self.neigh_.kneighbors(n_neighbors=self.n_neighbors,
                                             return_distance=True)
        dist = self._get_dist_by_method(dist_arr)

        self.decision_scores_ = dist.ravel()
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
        check_is_fitted(self, ['tree_', 'decision_scores_',
                               'threshold_', 'labels_'])

        X = check_array(X)

        # initialize the output score
        pred_scores = np.zeros([X.shape[0], 1])

        for i in range(X.shape[0]):
            x_i = X[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance of the current point
            dist_arr, _ = self.tree_.query(x_i, k=self.n_neighbors)
            dist = self._get_dist_by_method(dist_arr)
            pred_score_i = dist[-1]

            # record the current item
            pred_scores[i, :] = pred_score_i

        return pred_scores.ravel()

    def _get_dist_by_method(self, dist_arr):
        """Internal function to decide how to process passed in distance array

        Parameters
        ----------
        dist_arr : numpy array of shape (n_samples, n_neighbors)
            Distance matrix.

        Returns
        -------
        dist : numpy array of shape (n_samples,)
            The outlier scores by distance.
        """

        if self.method == 'largest':
            return dist_arr[:, -1]
        elif self.method == 'mean':
            return np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            return np.median(dist_arr, axis=1)

    def get_outlier_explainability_scores(self, ind, columns=None):
        """Compute per-feature outlier explainability scores.
        
        Calculates the average absolute distance to k-nearest neighbors
        for each feature dimension.
        
        Parameters
        ----------
        ind : int
            Index of the sample in training data.
            
        columns : list, optional (default=None)
            Specific feature indices. If None, use all features.
            
        Returns
        -------
        scores : numpy array of shape (n_features,)
            Average absolute distance to k-neighbors per dimension.
        """
        check_is_fitted(self, ['X_train_', 'tree_'])
        
        sample = self.X_train_[ind:ind+1, :]
        _, neighbor_indices = self.tree_.query(sample, k=self.n_neighbors)
        neighbors = self.X_train_[neighbor_indices[0], :]
        
        if columns is None:
            dim_distances = np.abs(neighbors - self.X_train_[ind, :])
        else:
            dim_distances = np.abs(neighbors[:, columns] - self.X_train_[ind, columns])
        
        return np.mean(dim_distances, axis=0)


    def explain_outlier(self, ind, columns=None, cutoffs=None,
                        feature_names=None, file_name=None,
                        file_type=None, max_features_per_plot=20,
                        compute_cutoffs=True):  # pragma: no cover
        """Plot dimensional outlier graph for a given data point.

        Parameters
        ----------
        ind : int
            The index of the data point to explain.

        columns : list, optional
            Specify a list of features/dimensions for plotting. If not 
            specified, use all features.
        
        cutoffs : list of floats in (0., 1), optional (default=[0.95, 0.99])
            The significance cutoff bands of the dimensional outlier graph.
        
        feature_names : list of strings, optional
            The display names of all columns of the dataset,
            to show on the y-axis of the plot.

        file_name : string, optional
            The name to save the figure.

        file_type : string, optional
            The file type to save the figure.

        max_features_per_plot : int, optional (default=20)
            Maximum number of features per plot. Splits into multiple plots if exceeded.

        compute_cutoffs : bool, optional (default=True)
            If True, computes dimensional scores for all samples to generate cutoff bands.
            If False, only computes dimensional score for the target sample (much faster).
            When True, results are cached for subsequent calls.

        Returns
        -------
        scores : numpy array
            The per-feature outlier scores for the specified sample.
        """
        check_is_fitted(self, ['X_train_', 'tree_', 'labels_'])
        
        if columns is None:
            columns = list(range(self.X_train_.shape[1]))
        
        cutoffs = [1 - self.contamination, 0.99] if cutoffs is None else cutoffs
        
        # Compute dimensional scores for target sample
        dim_scores = self.get_outlier_explainability_scores(ind, columns)
        
        # Compute cutoff bands if requested
        if compute_cutoffs:
            cache_key = tuple(columns)
            if cache_key in self._cached_dimensional_scores:
                all_scores = self._cached_dimensional_scores[cache_key]
            else:
                all_scores = np.zeros((self.X_train_.shape[0], len(columns)))
                for i in range(self.X_train_.shape[0]):
                    all_scores[i, :] = self.get_outlier_explainability_scores(i, columns)
                self._cached_dimensional_scores[cache_key] = all_scores
            
            cutoff_bands = {c: np.quantile(all_scores, q=c, axis=0) for c in cutoffs}
        else:
            cutoff_bands = None
        
        # Set feature names
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in columns]
        
        # Split into chunks if needed
        n_features = len(columns)
        chunks = []
        for start in range(0, n_features, max_features_per_plot):
            end = min(start + max_features_per_plot, n_features)
            chunk_cutoffs = None
            if cutoff_bands:
                chunk_cutoffs = {c: cutoff_bands[c][start:end] for c in cutoffs}
            chunks.append((columns[start:end], dim_scores[start:end], 
                        feature_names[start:end], chunk_cutoffs))
        
        # Plot each chunk
        for chunk_idx, (chunk_cols, chunk_scores, chunk_names, chunk_cutoffs) in enumerate(chunks):
            self._plot_explanation_chunk(ind, chunk_cols, chunk_scores, chunk_names, 
                                        chunk_cutoffs, chunk_idx, len(chunks), 
                                        file_name, file_type)
        
        return dim_scores


    def _plot_explanation_chunk(self, ind, columns, scores, names, cutoff_bands, 
                                idx, total, file_name, file_type):
        """Helper to plot one feature chunk."""
        n = len(columns)
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))
        y_pos = np.arange(n)
        
        # Determine colors based on cutoffs
        if cutoff_bands:
            cutoffs_sorted = sorted(cutoff_bands.keys())
            colors = []
            for i, score in enumerate(scores):
                if len(cutoffs_sorted) >= 2:
                    if score >= cutoff_bands[cutoffs_sorted[-1]][i]:
                        colors.append('#d62728')  # Red
                    elif score >= cutoff_bands[cutoffs_sorted[0]][i]:
                        colors.append('#ff7f0e')  # Orange
                    else:
                        colors.append('#1f77b4')  # Blue
                else:
                    colors.append('#d62728' if score >= cutoff_bands[cutoffs_sorted[0]][i] else '#1f77b4')
        else:
            colors = ['#000000'] * n  # Black when no cutoffs
        
        # Plot bars
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=0.5)
        
        # Add cutoff lines
        if cutoff_bands:
            cutoffs_sorted = sorted(cutoff_bands.keys())
            styles = ['--', '-.', ':']
            line_colors = ['#ff7f0e', '#d62728', '#8c564b']
            for cutoff, style, color in zip(cutoffs_sorted, styles, line_colors):
                for i, val in enumerate(cutoff_bands[cutoff]):
                    ax.plot([val, val], [i-0.4, i+0.4], style, color=color, linewidth=2,
                        label=f'{cutoff:.2f} Cutoff' if i == 0 else "")
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Average Distance to k-Neighbors', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        label = 'Outlier' if self.labels_[ind] == 1 else 'Inlier'
        overall_knn_score = self.decision_scores_[ind]
        
        if total > 1:
            title = (f'Outlier score breakdown for sample #{ind+1} ({label})\n'
                    f'k={self.n_neighbors} | Overall KNN={overall_knn_score:.3f} | '
                    f'Features {columns[0]+1}-{columns[-1]+1} (Part {idx+1}/{total})')
        else:
            title = (f'Outlier score breakdown for sample #{ind+1} ({label})\n'
                    f'k={self.n_neighbors}, method={self.method} | Overall KNN={overall_knn_score:.3f}')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {score:.3f}', ha='left', va='center', fontsize=9)
        
        # Legend
        if cutoff_bands:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.9)
        
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        plt.tight_layout()
        
        # Save file
        if file_name:
            name = f'{file_name}_part{idx+1}of{total}' if total > 1 else file_name
            if file_type:
                plt.savefig(f'{name}.{file_type}', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
        
        plt.show()