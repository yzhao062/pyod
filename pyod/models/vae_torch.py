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

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ..utils.stat_models import pairwise_distances_no_broadcast
from ..utils.utility import check_parameter


from torch import nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm 

class VAE_model(nn.Module):
    def __init__(self, 
                 n_features,
                 encoder_neurons,
                 decoder_neurons,
                 hidden_activation,
                 output_activation,
                 dropout_rate,
                 latent_dim,
                 device,
                 ):
        super(VAE_model, self).__init__()
        self.device = device
        # Input layer

        self.inputs = nn.Linear(n_features, n_features)
        self.act0 = hidden_activation()

        # Hidden layers
        encoder_hidden_list = []
        encoder_neurons = [n_features] + encoder_neurons
        for i in range(1, len(encoder_neurons)):
            encoder_hidden_list += [
                nn.Linear(encoder_neurons[i - 1], encoder_neurons[i]),
                nn.BatchNorm1d(encoder_neurons[i]),
                hidden_activation(),
                nn.Dropout(dropout_rate)
            ]
        self.encoder_hidden = nn.Sequential(*encoder_hidden_list)
        # Create mu and sigma of latent variables
        self.z_mean_layer = nn.Linear(encoder_neurons[-1], latent_dim)
        self.z_log_layer = nn.Linear(encoder_neurons[-1], latent_dim)


        # Latent input layer
        self.latent_input = nn.Linear(latent_dim, latent_dim)
        self.act1 = hidden_activation()

        # Hidden layers
        decoder_hidden_list = []
        decoder_neurons = [latent_dim] + decoder_neurons
        for i in range(1, len(decoder_neurons)):
            decoder_hidden_list += [
                nn.Linear(decoder_neurons[i - 1], decoder_neurons[i]),
                nn.BatchNorm1d(decoder_neurons[i]),
                hidden_activation(),
                nn.Dropout(dropout_rate)
            ]
        self.decoder_hidden = nn.Sequential(*decoder_hidden_list)
        
        # Output layer
        self.outputs = nn.Linear(decoder_neurons[-1], n_features)
        self.act2 = output_activation()

    def forward(self, X):
        x = self.act0(self.inputs(X))
        x = self.encoder_hidden(x)

        z_mean = self.z_mean_layer(x)
        z_log = self.z_log_layer(x)

        #sampling
        epsilon = torch.normal(0, 1, size=z_mean.shape).to(self.device)
        x = z_mean + torch.exp(0.5 * z_log) * epsilon

        x = self.act1(self.latent_input(x))
        x = self.decoder_hidden(x)

        out = self.act2(self.outputs(x))

        return out, z_mean, z_log



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

    hidden_activation : torch.nn Non-linear Activation instance, optional (default=Relu)
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    output_activation : torch.nn Non-linear Activation instance, optional (default=Sigmoid)
        Activation function to use for output layer.
        See https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    loss : torch.nn loss instance, optional (default=torch.nn.MSELoss)
        See https://pytorch.org/docs/stable/nn.html#loss-functions

    gamma : float, optional (default=1.0)
        Coefficient of beta VAE regime.
        Default is regular VAE.

    capacity : float, optional (default=0.0)
        Maximum capacity of a loss bottle neck.

    optimizer : torch.optim instance, optional (default=Adam)
        See https://pytorch.org/docs/stable/optim.html

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    weight_decay : float in (0., 1), optional (default=3e-4)
        The weight_decay parameter for torch.optim optimizer

    lr : float in (0., 1), optional (default=3e-4)
        The learning rate parameter for torch.optim optimizer

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        verbose mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = print losses

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

    device: str, optional (default=None)
        Device for torch model evaluating, if None, is chosen automatically from "cpu" and "cuda"

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : torch.nn.Module
        The underlying AutoEncoder in torch.

    history_: dict
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

    def __init__(self, encoder_neurons=None, decoder_neurons=None,
                 latent_dim=2, hidden_activation=nn.ReLU,
                 output_activation=nn.Sigmoid, loss=nn.MSELoss, optimizer=Adam, lr=3e-4,
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 weight_decay=3e-4, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1,
                 gamma=1.0, capacity=0.0, device="cpu"):
        super(VAE, self).__init__(contamination=contamination)
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.capacity = capacity

        self.optimizer = optimizer
        self.lr=lr

        if device is None:
            self.device = "cuda" if torch.cuda.is_avalible() else "cpu"
        else:
            self.device=device

        # default values
        if self.encoder_neurons is None:
            self.encoder_neurons = [128, 64, 32]

        if self.decoder_neurons is None:
            self.decoder_neurons = [32, 64, 128]

        self.encoder_neurons_ = self.encoder_neurons
        self.decoder_neurons_ = self.decoder_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)
        

    def vae_loss(self, inputs, outputs, z_mean, z_log):
        """ Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """

        reconstruction_loss = self.loss(inputs, outputs)
        reconstruction_loss *= self.n_features_
        kl_loss = 1 + z_log - torch.square(z_mean) - torch.exp(z_log)
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
        kl_loss = self.gamma * torch.abs(kl_loss - self.capacity)

        return torch.mean(reconstruction_loss + kl_loss)


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
        self.model_ = VAE_model(
            self.n_features_,
            self.encoder_neurons,
            self.decoder_neurons,
            self.hidden_activation,
            self.output_activation,
            self.dropout_rate,
            self.latent_dim,
            self.device
        ).to(self.device)
        
        self.optimizer = self.optimizer(self.model_.parameters(),  lr=self.lr, weight_decay=self.weight_decay)

        X_train, X_val = train_test_split(X, test_size=self.validation_size, shuffle=True)
        train_loader = DataLoader(X_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(X_val, batch_size=self.batch_size, shuffle=False)

        train_losses = []
        val_losses = []
        for epoch in (range(self.epochs)):
            train_epoch_loss = 0
            pbar_train = tqdm(train_loader, disable=(self.verbose == 0))
            for inputs in pbar_train:
                pbar_train.set_description(f"Train epoch {epoch}")
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device).float()

                outputs, z_mean, z_log = self.model_.forward(inputs)

                loss = self.vae_loss(inputs, outputs, z_mean, z_log)

                loss.backward()
                self.optimizer.step()

                train_epoch_loss += loss.item()
            train_losses.append(train_epoch_loss/len(train_loader))

            self.model_.eval()
            val_epoch_loss = 0
            pbar_val = tqdm(val_loader, disable=(self.verbose == 0))
            for inputs in pbar_val:
                pbar_val.set_description(f"Validation epoch {epoch}")
                inputs = inputs.to(self.device).float()
                with torch.no_grad():
                    outputs, z_mean, z_log = self.model_.forward(inputs)

                    loss = self.vae_loss(inputs, outputs, z_mean, z_log)
                    val_epoch_loss += loss.item()
            val_losses.append(val_epoch_loss/len(val_loader))
            if self.verbose == 2:
                print(f"Train loss on epoch {epoch}: {train_losses[-1]}, Val loss on epoch {epoch}: {val_losses[-1]}")
       
        self.history_ = {"train_loss" : train_losses, "val_loss" : val_losses}
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.forward(torch.Tensor(X_norm).float().to(self.device))[0].cpu().detach().numpy()
        self.decision_scores_ = pairwise_distances_no_broadcast(X_norm,
                                                                pred_scores)
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
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.forward(torch.Tensor(X_norm).float().to(self.device))[0].cpu().detach().numpy()
        return pairwise_distances_no_broadcast(X_norm, pred_scores)
