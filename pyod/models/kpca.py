# -*- coding: utf-8 -*-
"""Kernel Principal Component Analysis (KPCA) Outlier Detector
"""
# Author: Akira Tamamori <tamamori5917@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ..utils.utility import check_parameter


class PyODKernelPCA(KernelPCA):
    """A wrapper class for KernelPCA class of scikit-learn."""

    def __init__(
            self,
            n_components=None,
            kernel="rbf",
            gamma=None,
            degree=3,
            coef0=1,
            kernel_params=None,
            alpha=1.0,
            fit_inverse_transform=False,
            eigen_solver="auto",
            tol=0,
            max_iter=None,
            remove_zero_eig=False,
            copy_X=True,
            n_jobs=None,
            random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            remove_zero_eig=remove_zero_eig,
            n_jobs=n_jobs,
            copy_X=copy_X,
            random_state=check_random_state(random_state),
        )

    @property
    def get_centerer(self):
        """Return a protected member _centerer."""
        return self._centerer

    @property
    def get_kernel(self):
        """Return a protected member _get_kernel."""
        return self._get_kernel


class KPCA(BaseDetector):
    """KPCA class for outlier detection.

    PCA is performed on the feature space uniquely determined by the kernel,
    and the reconstruction error on the feature space is used as the anomaly score.

    See :cite:`hoffmann2007kernel`
    Heiko Hoffmann, "Kernel PCA for novelty detection,"
    Pattern Recognition, vol.40, no.3, pp. 863-874, 2007.
    https://www.sciencedirect.com/science/article/pii/S0031320306003414
    for details.

    Parameters
    ----------
    n_components : int, optional (default=None)
        Number of components. If None, all non-zero components are kept.

    n_selected_components : int, optional (default=None)
        Number of selected principal components
        for calculating the outlier scores. It is not necessarily equal to
        the total number of the principal components. If not set, use
        all principal components.

    kernel : string {'linear', 'poly', 'rbf', 'sigmoid',
                     'cosine', 'precomputed'}, optional (default='rbf')
        Kernel used for PCA.

    gamma : float, optional (default=None)
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.

    degree : int, optional (default=3)
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, optional (default=1)
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict, optional (default=None)
        Parameters (keyword arguments) and
        values for kernel passed as callable object.
        Ignored by other kernels.

    alpha : float, optional (default=1.0)
        Hyperparameter of the ridge regression that learns the
        inverse transform (when inverse_transform=True).

    eigen_solver : string, {'auto', 'dense', 'arpack', 'randomized'}, \
            default='auto'
        Select eigensolver to use. If `n_components` is much
        less than the number of training samples, randomized (or arpack to a
        smaller extend) may be more efficient than the dense eigensolver.
        Randomized SVD is performed according to the method of Halko et al.

        auto :
            the solver is selected by a default policy based on n_samples
            (the number of training samples) and `n_components`:
            if the number of components to extract is less than 10 (strict) and
            the number of samples is more than 200 (strict), the 'arpack'
            method is enabled. Otherwise the exact full eigenvalue
            decomposition is computed and optionally truncated afterwards
            ('dense' method).
        dense :
            run exact full eigenvalue decomposition calling the standard
            LAPACK solver via `scipy.linalg.eigh`, and select the components
            by postprocessing.
        arpack :
            run SVD truncated to n_components calling ARPACK solver using
            `scipy.sparse.linalg.eigsh`. It requires strictly
            0 < n_components < n_samples
        randomized :
            run randomized SVD.
            implementation selects eigenvalues based on their module; therefore
            using this method can lead to unexpected results if the kernel is
            not positive semi-definite.

    tol : float, optional (default=0)
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.

    max_iter : int, optional (default=None)
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.

    remove_zero_eig : bool, optional (default=False)
        If True, then all components with zero eigenvalues are removed, so
        that the number of components in the output may be < n_components
        (and sometimes even zero due to numerical instability).
        When n_components is None, this parameter is ignored and components
        with zero eigenvalues are removed regardless.

    copy_X : bool, optional (default=True)
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.

    n_jobs : int, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    sampling : bool, optional (default=False)
        If True, sampling subset from the dataset is performed only once,
        in order to reduce time complexity while keeping detection performance.

    subset_size : float in (0., 1.0) or int (0, n_samples), optional (default=20)
        If sampling is True, the size of subset is specified.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by np.random.

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

    def __init__(
            self,
            contamination=0.1,
            n_components=None,
            n_selected_components=None,
            kernel="rbf",
            gamma=None,
            degree=3,
            coef0=1,
            kernel_params=None,
            alpha=1.0,
            eigen_solver="auto",
            tol=0,
            max_iter=None,
            remove_zero_eig=False,
            copy_X=True,
            n_jobs=None,
            sampling=False,
            subset_size=20,
            random_state=None,
    ):
        super().__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.sampling = sampling
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)
        self.decision_scores_ = None
        self.n_selected_components_ = None

    def _check_subset_size(self, array):
        """Check subset size."""
        n_samples, _ = array.shape
        if isinstance(self.subset_size, int) is True:
            if 0 < self.subset_size <= n_samples:
                subset_size = self.subset_size
            else:
                raise ValueError(
                    f"subset_size={self.subset_size} "
                    f"must be between 0 and n_samples={n_samples}."
                )

        if isinstance(self.subset_size, float) is True:
            if 0.0 < self.subset_size <= 1.0:
                subset_size = int(self.subset_size * n_samples)
            else:
                raise ValueError("subset_size=%r must be between 0.0 and 1.0")

        return subset_size

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
        X = check_array(X, copy=self.copy_X)
        self._set_n_classes(y)

        # perform subsampling to reduce time complexity
        if self.sampling is True:
            subset_size = self._check_subset_size(X)
            random_indices = self.random_state.choice(
                X.shape[0],
                size=subset_size,
                replace=False,
            )
            X = X[random_indices, :]

        # copy the attributes from the sklearn Kernel PCA object
        if self.n_components is None:
            n_components = X.shape[0]  # use all dimensions
        else:
            if self.n_components < 1:
                raise ValueError(
                    f"`n_components` should be >= 1, got: {self.n_components}"
                )
            n_components = min(X.shape[0], self.n_components)

        # validate the number of components to be used for outlier detection
        if self.n_selected_components is None:
            self.n_selected_components_ = n_components
        else:
            self.n_selected_components_ = self.n_selected_components
        check_parameter(
            self.n_selected_components_,
            1,
            n_components,
            include_left=True,
            include_right=True,
            param_name="n_selected_components",
        )

        self.kpca = PyODKernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params,
            alpha=self.alpha,
            fit_inverse_transform=False,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            remove_zero_eig=self.remove_zero_eig,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        x_transformed = self.kpca.fit_transform(X)
        x_transformed = x_transformed[:, : self.n_selected_components_]

        centerer = self.kpca.get_centerer
        kernel = self.kpca.get_kernel

        potential = []
        for i in range(X.shape[0]):
            sample = X[i, :].reshape(1, -1)
            potential.append(kernel(sample))
        potential = np.array(potential).squeeze()
        potential = potential - 2 * centerer.K_fit_rows_ + centerer.K_fit_all_

        # reconstruction error
        self.decision_scores_ = potential - np.sum(np.square(x_transformed),
                                                   axis=1)
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
        check_is_fitted(self, ["decision_scores_", "threshold_", "labels_"])
        X = check_array(X)

        # Compute centered gram matrix between X and training data X_fit_
        centerer = self.kpca.get_centerer
        kernel = self.kpca.get_kernel
        gram_matrix = kernel(X, self.kpca.X_fit_)

        x_transformed = self.kpca.transform(X)
        x_transformed = x_transformed[:, : self.n_selected_components_]

        potential = []
        for i in range(X.shape[0]):
            sample = X[i, :].reshape(1, -1)
            potential.append(kernel(sample))
        potential = np.array(potential).squeeze()
        gram_fit_rows = np.sum(gram_matrix, axis=1) / gram_matrix.shape[1]
        potential = potential - 2 * gram_fit_rows + centerer.K_fit_all_

        # reconstruction error
        anomaly_scores = potential - np.sum(np.square(x_transformed), axis=1)

        return anomaly_scores
