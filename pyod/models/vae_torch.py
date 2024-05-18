# -*- coding: utf-8 -*-
"""
Using Variational Auto-Encoder for outlier detection (PyTorch)
"""
# Author: Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import torch
import torch.nn as nn
import tqdm

from .base import BaseDetector
from ..utils.torch_utility import PyODDataset, LinearBlock
from ..utils.stat_models import pairwise_distances_no_broadcast


class InnerVAE(nn.Module):
    def __init__(self,
                 n_features,
                 encoder_neurons=[128, 64, 32],
                 decoder_neurons=[32, 64, 128],
                 latent_dim=2,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 dropout_rate=0.2):
        super(InnerVAE, self).__init__()

        self.n_features = n_features
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.encoder_mu = nn.Linear(
            self.encoder_neurons[-1], self.latent_dim)
        self.encoder_logvar = nn.Linear(
            self.encoder_neurons[-1], self.latent_dim)

    def _build_encoder(self):
        encoder_layers = []
        encoder_layers.append(
            LinearBlock(
                self.n_features,
                self.encoder_neurons[0],
                activation_name=self.hidden_activation,
                dropout_rate=0
            )
        )
        for i in range(len(self.encoder_neurons) - 1):
            encoder_layers.append(
                LinearBlock(
                    self.encoder_neurons[i], 
                    self.encoder_neurons[i+1],
                    activation_name=self.hidden_activation,
                    dropout_rate=self.dropout_rate
                )
            )
        encoder = nn.Sequential(*encoder_layers)
        return encoder

    def _build_decoder(self):
        decoder_layers = []
        for i in range(len(self.decoder_neurons)):
            decoder_layers.append(
                LinearBlock(
                    self.decoder_neurons[i-1] \
                        if i > 0 else self.latent_dim,
                    self.decoder_neurons[i],
                    activation_name=self.hidden_activation,
                    dropout_rate=self.dropout_rate
                )
            )
        decoder_layers.append(
            LinearBlock(
                self.decoder_neurons[-1],
                self.n_features,
                activation_name=self.output_activation,
                dropout_rate=0
            )
        )
        decoder = nn.Sequential(*decoder_layers)
        return decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(mu.device)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        z_mu = self.encoder_mu(h)
        z_logvar = self.encoder_logvar(h)
        return z_mu, z_logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decode(z)
        return x_recon, z_mu, z_logvar


class VAE(BaseDetector):
    """ Variational auto encoder
    Encoder maps X onto a latent space Z
    Decoder samples Z from N(0,1)
    VAE_loss = Reconstruction_loss + KL_loss

    Reference
    See :cite:`kingma2013auto` Kingma, Diederik, Welling
    'Auto-Encodeing Variational Bayes'
    https://arxiv.org/abs/1312.6114 for details.

    beta VAE
    In Loss, the emphasis is on KL_loss
    and capacity of a bottleneck:
    VAE_loss = Reconstruction_loss + beta * KL_loss

    Reference
    See :cite:`burgess2018understanding` Burges et al
    'Understanding disentangling in beta-VAE'
    https://arxiv.org/pdf/1804.03599.pdf for details.

    Parameters
    ----------
    encoder_neurons : list, optional (default=[128, 64, 32])
        The number of neurons per hidden layer in encoder.

    decoder_neurons : list, optional (default=[32, 64, 128])
        The number of neurons per hidden layer in decoder.

    latent_dim : int, optional (default=2)

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://pytorch.org/docs/stable/nn.html for details.
        Currently only
        'relu': nn.ReLU()
        'sigmoid': nn.Sigmoid()
        'tanh': nn.Tanh()
        are supported. See pyod/utils/torch_utility.py for details.

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.

    loss_fn : obj, optional (default=torch.nn.MSELoss)
        Optimizer instance which implements torch.nn._Loss.
        One of https://pytorch.org/docs/stable/nn.html#loss-functions
        or a custom loss. Custom losses are currently unstable.

    beat : float, optional (default=1.0)
        Coefficient of beta VAE regime.
        Default is regular VAE.

    capacity : float, optional (default=0.0)
        Maximum capacity of a loss bottle neck.

    learning_rate : float, optional (default=1e-3)
        Learning rate for the optimizer. This learning_rate is given to
        an Adam optimizer (torch.optim.Adam).
        See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html for details.

    optimizer : obj, optional (default=torch.optim.Adam)

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    weight_decay : float, optional (default=1e-5)

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

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
        the proportion of outliers in the data set. When fitting this is
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
                 encoder_neurons=[128, 64, 32],
                 decoder_neurons=[32, 64, 128],
                 latent_dim=2,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss_fn=nn.MSELoss(),
                 beta=1.0,
                 capacity=0.0,
                 learning_rate=1e-3,
                 optimizer=torch.optim.Adam,
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 weight_decay=1e-5,
                 preprocessing=True,
                 verbose=1,
                 contamination=0.1,
                 device=None,
                 use_compile=False,
                 compile_mode='default'):
        super(VAE, self).__init__(contamination=contamination)

        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.beta = beta
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
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

        self.model = None
        self.best_model_dict = None
        self._mean = None
        self._std = None

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

        _, n_features = X.shape

        # conduct standardization if needed
        if self.preprocessing:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            train_set = PyODDataset(X, mean=self._mean, std=self._std)
        else:
            train_set = PyODDataset(X)

        # create data loader
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   drop_last=True)
        
        # build the model
        self.model = InnerVAE(n_features=n_features,
                              encoder_neurons=self.encoder_neurons,
                              decoder_neurons=self.decoder_neurons,
                              latent_dim=self.latent_dim,
                              hidden_activation=self.hidden_activation,
                              output_activation=self.output_activation,
                              dropout_rate=self.dropout_rate)
        self.model.to(self.device)

        # train the model
        self._train_vae(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

    def compute_loss(self, x, x_recon, mu, logvar):
        # reconstruction loss
        recon_loss = self.loss_fn(x_recon, x)
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # beta VAE
        kl_loss = self.beta * torch.abs(kl_loss - self.capacity)

        return recon_loss + kl_loss

    def _train_vae(self, train_loader):
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=self.weight_decay)

        # init the best model
        best_loss = float('inf')
        best_model_dict = None

        if self.use_compile:
            self.model = torch.compile(model=self.model, mode=self.compile_mode)
            print('Model compiled.')

        if self.verbose >= 1:
            print(self.model)

        for epoch in tqdm.trange(self.epochs, desc='VAE Training',
                                 disable=not self.verbose == 1):
            overall_loss = []
            for x in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                x_recon, mu, logvar = self.model(x)
                loss = self.compute_loss(x, x_recon, mu, logvar)
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
            epoch_loss = np.mean(overall_loss)
            if self.verbose >= 2:
                print(f"Epoch {epoch + 1} / {self.epochs}, Loss: {epoch_loss:.4f}")

            # track the best model
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
            dataset = PyODDataset(X, mean=self._mean, std=self._std)
        else:
            dataset = PyODDataset(X)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()

        outlier_scores = []
        with torch.no_grad():
            for x in data_loader:
                x_gpu = x.to(self.device)
                x_recon, _, _ = self.model(x_gpu)
                outlier_scores.append(
                    pairwise_distances_no_broadcast(
                        x.numpy(),
                        x_recon.cpu().detach().numpy())
                )
        return np.concatenate(outlier_scores)
