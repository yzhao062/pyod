# -*- coding: utf-8 -*-
"""Deep Semi-Supervised Anomaly Detection (Deep SAD) for outlier detection
"""
# Author: Jayesh Suryavanshi <jayeshsuryavanshi808@gmail.com>
# License: BSD 2 clause
#
# The Deep SAD objective and the hypersphere-center initialization are
# adapted from the authors' reference implementation
# https://github.com/lukasruff/Deep-SAD-PyTorch (MIT license) and from
# DeepOD's ``deepod/models/tabular/dsad.py`` (BSD-3 license). Neither is
# copied verbatim; the code below follows PyOD conventions and mirrors
# the structure of ``pyod/models/deep_svdd.py``.


import copy

import numpy as np

try:
    import torch
except ImportError:
    print('please install torch first')

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_random_state
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDetector
from ..utils.torch_utility import get_activation_by_name
from ..utils.utility import check_parameter

optimizer_dict = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta,
    'adamw': optim.AdamW,
    'nadam': optim.NAdam,
    'asgd': optim.ASGD
}


class InnerDeepSAD(nn.Module):
    """Inner encoder network for the Deep SAD model.

    The network maps the input into a low-dimensional representation space
    in which the semi-supervised hypersphere objective is optimized. Unlike
    Deep SVDD, no autoencoder branch is used: the Deep SAD loss operates
    purely on the distance of the representation to the hypersphere center.

    Parameters
    ----------
    n_features : int
        Number of features in the input data.

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. The last entry is the
        dimensionality of the representation space. A single entry
        ``[k]`` maps the input straight to the representation space
        with one bias-free linear layer.

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.

    dropout_rate : float in [0., 1), optional (default=0.2)
        The dropout applied after the activation of every hidden layer
        (``hidden_neurons[:-1]``). The final representation layer
        ``hidden_neurons[-1]`` has no activation and no dropout.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used.
    """

    def __init__(self, n_features, hidden_neurons, hidden_activation,
                 dropout_rate, l2_regularizer):
        super(InnerDeepSAD, self).__init__()
        self.n_features = n_features
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.model = self._build_model()

    def _init_c(self, X_norm, eps=0.1):
        """Initialize the hypersphere center as the mean of an initial
        forward pass. Center components that are too close to zero are
        pushed away by ``eps`` so that they cannot be trivially matched
        by zero network weights.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_norm)
        c = torch.mean(out, dim=0)
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c > 0)] = eps
        return c

    def _build_model(self):
        layers = nn.Sequential()
        if len(self.hidden_neurons) == 1:
            layers.add_module('net_output',
                              nn.Linear(self.n_features,
                                        self.hidden_neurons[0], bias=False))
            return layers
        layers.add_module('input_layer',
                          nn.Linear(self.n_features, self.hidden_neurons[0],
                                    bias=False))
        layers.add_module('hidden_activation_e0',
                          get_activation_by_name(self.hidden_activation))
        layers.add_module('hidden_dropout_e0', nn.Dropout(self.dropout_rate))
        for i in range(1, len(self.hidden_neurons) - 1):
            layers.add_module(f'hidden_layer_e{i}',
                              nn.Linear(self.hidden_neurons[i - 1],
                                        self.hidden_neurons[i], bias=False))
            layers.add_module(f'hidden_activation_e{i}',
                              get_activation_by_name(self.hidden_activation))
            layers.add_module(f'hidden_dropout_e{i}',
                              nn.Dropout(self.dropout_rate))
        layers.add_module('net_output',
                          nn.Linear(self.hidden_neurons[-2],
                                    self.hidden_neurons[-1], bias=False))
        return layers

    def forward(self, x):
        return self.model(x)


class DeepSAD(BaseDetector):
    """Deep Semi-Supervised Anomaly Detection (Deep SAD) is the
    semi-supervised extension of Deep SVDD. A neural network is trained to
    map the data into a representation space where inliers are pulled
    toward a fixed hypersphere center ``c`` while labeled anomalies are
    pushed away from it. Unlabeled samples fall back on the unsupervised
    Deep SVDD objective (minimize the squared distance to ``c``); labeled
    anomalies use the inverse-distance term, so their contribution grows
    as they approach the center. The distance to ``c`` is used as the
    outlier score at inference time.
    See :cite:`ruff2020deepsad` for details.

    Parameters
    ----------
    n_features : int
        Number of features in the input data.

    c : float, numpy array, or torch tensor, optional (default=None)
        Deep SAD hypersphere center. When None (the default) the center
        is initialized from the first forward pass of the freshly built
        network; set ``random_state`` for repeatable results. A scalar is
        expanded to the representation width ``hidden_neurons[-1]``; a
        vector must match that width. The value must be finite and not
        all zeros: an all-zero center is the trivial solution of the
        objective for a bias-free network and collapses the hypersphere.
        The fitted center is exposed as ``c_``.

    eta : float, optional (default=1.0)
        Weight of the labeled term in the Deep SAD loss. Must be positive:
        a non-positive weight would reward moving labeled anomalies toward
        the center. Higher values place more emphasis on the (few) labeled
        anomalies relative to the unlabeled samples.

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. The last entry is the
        dimensionality of the representation space, e.g. [64, 32]. A
        single entry, e.g. [32], maps the input straight to the
        representation space with one bias-free linear layer.

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    optimizer : str, optional (default='adam')
        Name of the optimizer. One of 'sgd', 'adam', 'rmsprop',
        'adagrad', 'adadelta', 'adamw', 'nadam' or 'asgd'.

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in [0., 1), optional (default=0.2)
        The dropout applied after the activation of every hidden layer
        (``hidden_neurons[:-1]``). The final representation layer
        ``hidden_neurons[-1]`` has no activation and no dropout.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in [0., 1), optional (default=0.1)
        The fraction of the training samples (and their labels) held out
        as a validation set, drawn with ``random_state`` and stratified
        on the semi-supervised targets: the unlabeled samples and the
        labeled anomalies are split separately, each holding out
        ``round(n * validation_size)`` of its ``n`` members but never all
        of them, so a lone labeled anomaly always stays in the training
        split. The scaler is fitted on, the center is initialized from and the
        network is trained on the remaining samples; after every epoch
        the Deep SAD loss is computed on the held-out samples and the
        weights of the epoch with the lowest validation loss are restored
        at the end of training. Set to 0 to train on every sample and
        select the best epoch by training loss.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    eps : float, optional (default=1e-6)
        A small value added to the distance of labeled anomalies for
        numerical stability in the inverse-distance term. Must be a
        positive finite float.

    verbose : int, optional (default=1)
        Verbosity mode.
        - 0 = silent
        - 1 = progress bar

    random_state : random_state: int, RandomState instance or None,
        optional (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, an integer seed is
        drawn from it (advancing it) to seed torch; If None, the random
        number generator is the RandomState instance used by
        `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is
        used to define the threshold on the decision function.

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

    c_ : torch tensor of shape (hidden_neurons[-1],)
        The fitted hypersphere center used by the training objective and
        by ``decision_function``. Either the center initialized from the
        data (when ``c`` is None) or a validated copy of the
        user-supplied ``c``.
    """

    def __init__(self, n_features, c=None, eta=1.0, hidden_neurons=None,
                 hidden_activation='relu', optimizer='adam', epochs=100,
                 batch_size=32, dropout_rate=0.2, l2_regularizer=0.1,
                 validation_size=0.1, preprocessing=True, eps=1e-6,
                 verbose=1, random_state=None, contamination=0.1):
        super(DeepSAD, self).__init__(contamination=contamination)

        self.n_features = n_features
        self.c = c
        self.eta = eta
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.eps = eps
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None
        self.best_model_dict = None

        if self.random_state is not None:
            if isinstance(self.random_state, (int, np.integer)):
                torch_seed = self.random_state
            else:
                torch_seed = int(
                    check_random_state(self.random_state).randint(
                        0, np.iinfo(np.int32).max))
            torch.manual_seed(torch_seed)
        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)
        check_parameter(eta, 0, np.inf, param_name='eta')
        check_parameter(validation_size, 0, 1, param_name='validation_size',
                        include_left=True)
        check_parameter(eps, 0, np.inf, param_name='eps')
        if not np.isfinite(eps):
            raise ValueError(
                'eps is set to {}. Must be a positive finite float.'.format(
                    eps))

    def _process_semi_targets(self, y, n_samples):
        """Turn the optional labels ``y`` into Deep SAD semi-supervised
        targets in ``{-1, 0}``.

        - Unlabeled / normal samples receive a target of ``0`` and are
          trained with the unsupervised Deep SVDD objective (minimize the
          squared distance to the center).
        - Labeled anomalies (``y == 1``) receive a target of ``-1`` and
          are trained with the inverse-distance term, which pushes them
          away from the center.

        Passing ``y=None`` makes every sample unlabeled, recovering the
        unsupervised Deep SVDD behavior.
        """
        semi_targets = np.zeros(n_samples, dtype=np.float32)
        if y is not None:
            y = np.asarray(y).ravel()
            semi_targets[y == 1] = -1
        return semi_targets

    def _split_validation(self, semi_targets):
        """Split the sample indices into the training and the validation
        fold, stratified on the semi-supervised targets. ``round(n *
        validation_size)`` samples of each target class are held out,
        but never all of them, so a lone labeled anomaly always stays in
        the training fold.
        """
        n_samples = len(semi_targets)
        if self.validation_size == 0:
            return np.arange(n_samples), np.arange(0)
        random_state = check_random_state(self.random_state)
        fit_idx, val_idx = [], []
        for target in np.unique(semi_targets):
            idx = np.flatnonzero(semi_targets == target)
            idx = idx[random_state.permutation(len(idx))]
            n_val = min(int(round(len(idx) * self.validation_size)),
                        len(idx) - 1)
            val_idx.append(idx[:n_val])
            fit_idx.append(idx[n_val:])
        fit_idx = np.sort(np.concatenate(fit_idx))
        val_idx = np.sort(np.concatenate(val_idx))
        return fit_idx, val_idx

    def _deep_sad_loss(self, outputs, semi_targets):
        """Deep SAD loss: unlabeled/normal points (target ``0``) minimize
        the squared distance to the center; labeled anomalies (target
        ``-1``) minimize the inverse distance, pushing them away from
        ``c``.
        """
        dist = torch.sum((outputs - self.c_) ** 2, dim=-1)
        losses = torch.where(semi_targets == 0, dist,
                             self.eta * ((dist + self.eps) ** semi_targets))
        return torch.mean(losses)

    def fit(self, X, y=None):
        """Fit detector. Deep SAD is semi-supervised: ``y`` is optional
        and, when provided, marks labeled anomalies (``y == 1``). All other
        samples are treated as unlabeled and use the unsupervised objective.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            Optional labels, where 1 indicates a known anomaly and 0 marks
            an unlabeled/normal sample. If None, every sample is unlabeled.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        # Deep SAD is semi-supervised, so y carries partial supervision and
        # is expected; treat this as a binary anomaly-detection problem.
        self._classes = 2

        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        semi_targets = self._process_semi_targets(y, self.n_samples_)

        # Hold out the validation split used for best-epoch selection; the
        # scaler, the center and the training batches only see the
        # remaining samples.
        fit_idx, val_idx = self._split_validation(semi_targets)
        X_fit, semi_fit = X[fit_idx], semi_targets[fit_idx]
        if len(val_idx) > 0:
            X_val, semi_val = X[val_idx], semi_targets[val_idx]
        else:
            X_val, semi_val = None, None

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_fit = self.scaler_.fit_transform(X_fit)
            if X_val is not None:
                X_val = self.scaler_.transform(X_val)

        # Build Deep SAD model & fit with X. Unlike Deep SVDD (which
        # reverses the encoder in an optional autoencoder), Deep SAD uses
        # only the encoder, so hidden layers may be wider than the input.
        self.model_ = InnerDeepSAD(self.n_features,
                                   hidden_neurons=self.hidden_neurons,
                                   hidden_activation=self.hidden_activation,
                                   dropout_rate=self.dropout_rate,
                                   l2_regularizer=self.l2_regularizer)

        X_fit = torch.tensor(X_fit, dtype=torch.float32)
        semi_fit = torch.tensor(semi_fit, dtype=torch.float32)
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            semi_val = torch.tensor(semi_val, dtype=torch.float32)

        # Initialize the hypersphere center from a first forward pass, or
        # validate the user-supplied center against the representation
        # space.
        rep_dim = self.hidden_neurons[-1]
        if self.c is None:
            center = self.model_._init_c(X_fit)
        else:
            center = torch.as_tensor(self.c, dtype=X_fit.dtype,
                                     device=X_fit.device)
        if center.ndim == 0:
            center = center.repeat(rep_dim)
        elif center.ndim != 1 or center.numel() != rep_dim:
            raise ValueError(
                "c must be a scalar or a vector of length {} "
                "(hidden_neurons[-1]), got shape {}".format(
                    rep_dim, tuple(center.shape)))
        if not torch.isfinite(center).all():
            raise ValueError("c must contain only finite values.")
        if not torch.any(center != 0):
            raise ValueError(
                "c is all zeros, which makes the trivial all-zero-weights "
                "solution optimal and collapses the hypersphere. Pass "
                "c=None to initialize the center from the data, or "
                "supply a non-zero center.")
        self.c_ = center.detach().clone()

        dataset = TensorDataset(X_fit, semi_fit)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True)

        best_loss = float('inf')
        best_model_dict = None

        optimizer = optimizer_dict[self.optimizer](
            self.model_.parameters(), weight_decay=self.l2_regularizer)

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0
            for batch_x, batch_semi in dataloader:
                optimizer.zero_grad()
                loss = self._deep_sad_loss(self.model_(batch_x), batch_semi)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            epoch_loss /= len(dataset)
            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    selection_loss = self._deep_sad_loss(
                        self.model_(X_val), semi_val).item()
            else:
                selection_loss = epoch_loss
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_model_dict = copy.deepcopy(self.model_.state_dict())
            if self.verbose:
                message = (f"Epoch {epoch + 1}/{self.epochs}, "
                           f"Loss: {epoch_loss}")
                if X_val is not None:
                    message += f", Validation loss: {selection_loss}"
                print(message)
        self.best_model_dict = best_model_dict
        if self.best_model_dict is not None:
            self.model_.load_state_dict(self.best_model_dict)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score is the squared distance of the learned
        representation to the hypersphere center. For consistency, outliers
        are assigned with larger anomaly scores.

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
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_norm)
            dist = torch.sum((outputs - self.c_) ** 2, dim=-1)
        anomaly_scores = dist.numpy()
        return anomaly_scores
