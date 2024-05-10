# -*- coding: utf-8 -*-

"""Variational Auto Encoder (VAE)
and beta-VAE for Unsupervised Outlier Detection

Reference:
        :cite:`kingma2013auto` Kingma, Diederik, Welling
        'Auto-Encodeing Variational Bayes'
        https://arxiv.org/abs/1312.6114
        
        :cite:`burgess2018understanding` Burges et al
        'Understanding disentangling in beta-VAE'
        https://arxiv.org/pdf/1804.03599.pdf
"""

# Author: Andrij Vasylenko <andrij@liverpool.ac.uk>
# License: BSD 2 clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import chain

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from .base_dl import _get_tensorflow_version
from ..utils.stat_models import pairwise_distances_no_broadcast
from ..utils.utility import check_parameter

from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import optim
from tqdm import tqdm


_activation_classes = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

def _resolve_activation(activation):
    if isinstance(activation, str) and activation in _activation_classes:
        return _activation_classes[activation]
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f'Activation must be nn.Module subclass or one of {_activation_classes.keys()}')
    
def _resolve_loss(loss):
    if loss == 'mse':
        return nn.MSELoss
    elif isinstance(loss, callable):
        return loss
    else:
        raise ValueError(f'Loss must be "mse" or some callable')
    
def _resolve_optim(optimizer):
    if optimizer == 'adam':
        return optim.Adam
    else:
        raise ValueError(f'Only "adam" is supported as optimizer')

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
    VAE_loss = Reconstruction_loss + gamma*KL_loss

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

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    loss : str or obj, optional (default=keras.losses.mean_squared_error
        String (name of objective function) or objective function.
        See https://keras.io/losses/

    gamma : float, optional (default=1.0)
        Coefficient of beta VAE regime.
        Default is regular VAE.

    capacity : float, optional (default=0.0)
        Maximum capacity of a loss bottle neck.

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        See https://keras.io/optimizers/

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        verbose mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, opti
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the r
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is
        to define the threshold on the decision function.
        
    device : str
        Torch device to train and inference autoencoder

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    encoder_ : nn.Module
        The underlying AutoEncoder Encoder
        
    decoder_ : nn.Module
        The underlying AutoEncoder Decoder

    history_: List[float]
        The AutoEncoder val losses for every epoch

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

    def __init__(self, encoder_neurons=None, decoder_neurons=None,
                 latent_dim=2, hidden_activation='relu',
                 output_activation='sigmoid', loss='mse', optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=42, contamination=0.1,
                 gamma=1.0, capacity=0.0, device='cpu'):
        super(VAE, self).__init__(contamination=contamination)
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.hidden_activation = hidden_activation
        self.hidden_activation_cls = _resolve_activation(hidden_activation)
        self.output_activation = output_activation
        self.output_activation_cls = _resolve_activation(output_activation)
        self.loss = loss
        self.loss_fn = _resolve_loss(loss)()
        self.optimizer = optimizer
        self.optimizer_cls = _resolve_optim(optimizer)
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.capacity = capacity
        self.device = device

        # default values
        if self.encoder_neurons is None:
            self.encoder_neurons = [128, 64, 32]

        if self.decoder_neurons is None:
            self.decoder_neurons = [32, 64, 128]

        self.encoder_neurons_ = self.encoder_neurons
        self.decoder_neurons_ = self.decoder_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def _vae_loss(self, inputs, outputs, z_mean, z_log):
        """ Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """

        reconstruction_loss = self.loss_fn(inputs, outputs)
        reconstruction_loss *= self.n_features_
        kl_loss = 1 + z_log - z_mean ** 2 - torch.exp(z_log)
        kl_loss = -0.5 * kl_loss.sum(axis=-1)
        kl_loss = self.gamma * (kl_loss - self.capacity).abs()

        return (reconstruction_loss + kl_loss).mean()
    
    def _forward_encoder(self, X):
        is_encoder_input_layer = True
        activity_regularization = 0
        for layer in self.encoder_:
            X = layer(X)
            if isinstance(layer, nn.Linear):
                activity_regularization = (X ** 2).sum() + activity_regularization
        return X, activity_regularization
        
    def _forward_vae(self, X):
        X, activity_regularization = self._forward_encoder(X)
        latent_stats = X.reshape(-1, 2, self.latent_dim)
        z_mean = latent_stats[:, 0]
        z_log = latent_stats[:, 1]
        
        # reparametrization trick
        epsilon = torch.randn_like(z_mean)  # mean=0, std=1.0
        sampled_latent = z_mean + torch.exp(0.5 * z_log) * epsilon

        return self.decoder_(sampled_latent), z_mean, z_log, activity_regularization

    def _build_model(self):
        """Build VAE = encoder + decoder"""

        # Build Encoder
        encoder = []
        # Input layer
        encoder.append(nn.Linear(self.n_features_, self.n_features_))
        encoder.append(self.hidden_activation_cls())
        # Hidden layers
        prev_neurons = self.n_features_
        for neurons in self.encoder_neurons:
            # TODO add activation regularizer
            encoder.append(nn.Linear(prev_neurons, neurons))
            encoder.append(self.hidden_activation_cls())
            encoder.append(nn.Dropout(self.dropout_rate))
            prev_neurons = neurons
        # Create mu and sigma of latent variables
        encoder.append(nn.Linear(prev_neurons, 2 * self.latent_dim))
        encoder = nn.Sequential(*encoder)
        encoder.to(self.device)
        
        if self.verbose >= 1:
            print(encoder)
        
        decoder = []
        # Latent input layer
        decoder.append(nn.Linear(self.latent_dim, self.latent_dim))
        decoder.append(self.hidden_activation_cls())
        # Hidden layers
        prev_neurons = self.latent_dim
        for neurons in self.decoder_neurons:
            decoder.append(nn.Linear(prev_neurons, neurons))
            decoder.append(self.hidden_activation_cls())
            decoder.append(nn.Dropout(self.dropout_rate))
            prev_neurons = neurons
        # Create mu and sigma of latent variables
        decoder.append(nn.Linear(prev_neurons, self.n_features_))
        decoder.append(self.output_activation_cls())
        decoder = nn.Sequential(*decoder)
        decoder.to(self.device)
        
        return encoder, decoder      
    
    def _fit_vae(self, X, epochs, batch_size, shuffle, validation_split, verbose=False):
        dataset = TensorDataset(X)
        
        generator = torch.Generator().manual_seed(self.random_state)
        train_ds, val_ds = random_split(dataset, [1-validation_split, validation_split], generator)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        optimizer = self.optimizer_cls(chain(self.encoder_.parameters(), self.decoder_.parameters()))
        val_losses = []
        
        for i in range(epochs):
            iterable = tqdm(train_dl) if verbose else train_dl
            for X_batch in iterable:
                X_batch = X_batch[0].to(self.device)
                X_batch = X_batch.float()
                X_corrupted, z_mean, z_log, activ_reg = self._forward_vae(X_batch)
                loss = self._vae_loss(X_batch, X_corrupted, z_mean, z_log)
                # Add activity regularization from encoder Linear layers
                # Regularization is divided by batch size to conform with Keras
                # https://keras.io/api/layers/regularizers/
                loss += activ_reg * self.l2_regularizer / X_batch.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            iterable = tqdm(val_dl) if verbose else val_dl
            losses = []
            for X_batch in iterable:
                X_batch = X_batch[0].to(self.device)
                X_batch = X_batch.float()
                with torch.no_grad():
                    X_corrupted, z_mean, z_log, activ_reg = self._forward_vae(X_batch)
                    loss = self._vae_loss(X_batch, X_corrupted, z_mean, z_log)
                    loss += activ_reg * self.l2_regularizer / X_batch.shape[0]
                    losses.append(loss.item())
            val_losses.append(np.mean(losses))
            print(f'Epoch: {i}, Val loss: {val_losses[-1]}')
            
        return val_losses
        
    def fit(self, X, y=None):
        """Fit detector. y is optional for unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Validate and complete the number of hidden neurons
        if np.min(self.encoder_neurons) > self.n_features_:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")

        # Build VAE model & fit with X
        self.encoder_, self.decoder_ = self._build_model()
        self.history_ = self._fit_vae(torch.from_numpy(X_norm),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    validation_split=self.validation_size,
                    verbose=self.verbose)
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        self.decision_scores_ = self.decision_function(X)
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
        check_is_fitted(self, ['encoder_', 'decoder_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        X_norm_pt = torch.from_numpy(X_norm).float().to(self.device)
        with torch.no_grad():
            pred_scores, _, _, _ = self._forward_vae(X_norm_pt)
        pred_scores = pred_scores.cpu().numpy()
        return pairwise_distances_no_broadcast(X_norm, pred_scores)
