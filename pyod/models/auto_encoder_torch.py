# -*- coding: utf-8 -*-
"""Using AutoEncoder with Outlier Detection (PyTorch)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>, Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import torch
from torch import nn
import tqdm

from .base import BaseDetector
from ..utils.torch_utility import PyODDataset, LinearBlock
from ..utils.stat_models import pairwise_distances_no_broadcast


class InnerAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=[128, 64],
                 dropout_rate=0.2,
                 batch_norm=True,
                 hidden_activation='relu'):
        super(InnerAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder_layers = []
        encoder_layers.append(
            LinearBlock(self.n_features, self.hidden_neurons[0],
                        activation_name=self.hidden_activation,
                        batch_norm=self.batch_norm,
                        dropout_rate=self.dropout_rate)
        )
        for i in range(1, len(self.hidden_neurons)):
            encoder_layers.append(
                LinearBlock(self.hidden_neurons[i - 1], self.hidden_neurons[i],
                            activation_name=self.hidden_activation,
                            batch_norm=self.batch_norm,
                            dropout_rate=self.dropout_rate)
            )
        encoder = nn.Sequential(*encoder_layers)
        return encoder

    def _build_decoder(self):
        decoder_layers = []
        for i in range(len(self.hidden_neurons) - 1, 0, -1):
            decoder_layers.append(
                LinearBlock(self.hidden_neurons[i], self.hidden_neurons[i - 1],
                            activation_name=self.hidden_activation,
                            batch_norm=self.batch_norm,
                            dropout_rate=self.dropout_rate)
            )
        decoder_layers.append(
            LinearBlock(self.hidden_neurons[0], self.n_features,
                        activation_name='sigmoid',
                        batch_norm=self.batch_norm,
                        dropout_rate=0)
        )
        decoder = nn.Sequential(*decoder_layers)
        return decoder

    def forward(self, x):
        # we could return the latent representation here after the encoder
        # as the latent representation
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoder(BaseDetector):
    """Auto Encoder (AE) is a type of neural networks for learning useful data
    representations in an unsupervised manner. Similar to PCA, AE could be used
    to detect outlying objects in the data by calculating the reconstruction
    errors. See :cite:`aggarwal2015outlier` Chapter 3 for details.

    Notes
    -----
        This is the PyTorch version of AutoEncoder. See auto_encoder.py for
        the TensorFlow version.

    Parameters
    ----------
    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. So the network has the
        structure as [n_features, 64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://pytorch.org/docs/stable/nn.html for details.
        'elu', 'leaky_relu', 'relu', 'sigmoid', 'softmax', 'softplus', 'tanh'
        are supported. See pyod/utils/torch_utility.py for details.

    batch_norm : boolean, optional (default=True)
        Whether to apply Batch Normalization,
        See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

    learning_rate : float, optional (default=1e-3)
        Learning rate for the optimizer. This learning_rate is given to
        an Adam optimizer (torch.optim.Adam).
        See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    weight_decay : float, optional (default=1e-5)
        The weight decay for Adam optimizer.
        See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    loss_fn : obj, optional (default=torch.nn.MSELoss)
        Optimizer instance which implements torch.nn._Loss.
        One of https://pytorch.org/docs/stable/nn.html#loss-functions
        or a custom loss. Custom losses are currently unstable.

    verbose : int, optional (default=1)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.
        !CURRENTLY NOT SUPPORTED.!

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    device : str, optional (default=None)
        The device to use for the model. If None, it will be decided
        automatically. If you want to use MPS, set it to 'mps'.

    use_compile : bool, optional (default=False)
        Whether to compile the model. If True, the model will be compiled before training. 
        This is only available for PyTorch version >= 2.0.0. and Python < 3.12.

    compile_mode : str, optional (default='default')
        The mode to compile the model. 
        Can be either “default”, “reduce-overhead”, “max-autotune” or “max-autotune-no-cudagraphs”.
        See https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile for details.

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
        The AutoEncoder training history.

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

    def __init__(self,
                 hidden_neurons=[64, 32],
                 hidden_activation='relu',
                 batch_norm=True,
                 learning_rate=1e-3,
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 weight_decay=1e-5,
                 preprocessing=True,
                 loss_fn=nn.MSELoss(),
                 verbose=1,
                 contamination=0.1,
                 device=None,
                 use_compile=False,
                 compile_mode='default'):
        super(AutoEncoder, self).__init__(contamination=contamination)

        # save the initialization values
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.device = device
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            # If you want to use MPS, uncomment the following lines
            # self.device = torch.device(
            #     "mps" if torch.backends.mps.is_available() else self.device)
        self.use_compile = use_compile
        self.compile_mode = compile_mode

        # initialize the model
        self.model = None
        self.best_model_dict = None

    # noinspection PyUnresolvedReferences
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

        _, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X=X)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   drop_last=True)

        # initialize the model
        self.model = InnerAutoencoder(
            n_features=n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)

        # move to device and print model information
        self.model.to(self.device)

        # train the autoencoder to find the best one
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

    def _train_autoencoder(self, train_loader):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        best_loss = float('inf')
        best_model_dict = None

        if self.use_compile:
            self.model = torch.compile(model=self.model, mode=self.compile_mode)
            print('Model compiled.')

        if self.verbose >= 1:
            print(self.model)

        for epoch in tqdm.trange(self.epochs, desc='AutoEncoder Training',
                                 disable=not self.verbose==1):
            overall_loss = []
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, data)
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
            epoch_loss = np.mean(overall_loss)
            if self.verbose == 2:
                print(f'Epoch {epoch + 1}/{self.epochs} Loss: {epoch_loss:.4f}')

            # track the best model so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_dict = self.model.state_dict()
        self.best_model_dict = best_model_dict

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
        check_is_fitted(self, ['model', 'best_model_dict'])
        self.model.load_state_dict(self.best_model_dict)
        X = check_array(X)

        if self.preprocessing:
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()

        outlier_scores = []
        with torch.no_grad():
            for data in dataloader:
                data_gpu = data.to(self.device).float()
                outlier_scores.append(
                    pairwise_distances_no_broadcast(
                        data.cpu().numpy(),
                        self.model(data_gpu).cpu().numpy())
                )
        return np.concatenate(outlier_scores)
