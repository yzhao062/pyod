# -*- coding: utf-8 -*-
"""Anomaly Detection with Generative Adversarial Networks  (AnoGAN)
 Paper: https://arxiv.org/pdf/1703.05921.pdf
 Note, that this is another implementation of AnoGAN as the one from https://github.com/fuchami/ANOGAN
"""
# Author: Michiel Bongaerts (but not author of the AnoGAN method)
# License: BSD 2 clause

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    print('please install torch first')

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDetector
from ..utils.torch_utility import get_activation_by_name
from ..utils.utility import check_parameter


class Generator(nn.Module):
    def __init__(self, latent_dim_G, n_features, G_layers, dropout_rate,
                 activation_hidden, output_activation):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim_G
        self.n_features = n_features
        self.layers = G_layers
        self.dropout_rate = dropout_rate
        self.activation_hidden = activation_hidden
        self.output_activation = output_activation

        self.model = self._build_generator()

    def _build_generator(self):
        layers = [nn.Dropout(self.dropout_rate), nn.Linear(
            self.latent_dim, self.layers[0]),
                  get_activation_by_name(self.activation_hidden)]
        for i in range(1, len(self.layers)):
            layers.extend([
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.layers[i - 1], self.layers[i]),
                get_activation_by_name(self.activation_hidden)
            ])
        layers.append(nn.Linear(self.layers[-1], self.n_features))
        if self.output_activation:
            layers.append(get_activation_by_name(self.output_activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, n_features, D_layers, dropout_rate, activation_hidden):
        super(Discriminator, self).__init__()
        self.n_features = n_features
        self.layers = D_layers
        self.dropout_rate = dropout_rate
        self.activation_hidden = activation_hidden

        self.model = self._build_discriminator()

    def _build_discriminator(self):
        layers = [nn.Dropout(self.dropout_rate), nn.Linear(
            self.n_features, self.layers[0]),
                  get_activation_by_name(self.activation_hidden)]
        for i in range(1, len(self.layers)):
            layers.extend([
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.layers[i - 1], self.layers[i]),
                get_activation_by_name(self.activation_hidden)
            ])
        layers.extend([nn.Linear(self.layers[-1], 1), nn.Sigmoid()])
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class QueryModel(nn.Module):
    def __init__(self, generator, discriminator, latent_dim_G):
        super(QueryModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.z_gamma_layer = nn.Linear(latent_dim_G, latent_dim_G)

    def forward(self, query_sample):
        z_gamma = self.z_gamma_layer(query_sample)
        sample_gen = self.generator(z_gamma)
        sample_disc_latent = self.discriminator(sample_gen)
        return z_gamma, sample_gen, sample_disc_latent


class AnoGAN(BaseDetector):
    """Anomaly Detection with Generative Adversarial Networks  (AnoGAN).
    See the original paper "Unsupervised anomaly detection with generative
    adversarial networks to guide marker discovery".

    See :cite:`schlegl2017unsupervised` for details.

    Parameters
    ----------

    output_activation : str, optional (default=None)
        Activation function to use for output layer.


    activation_hidden : str, optional (default='tanh')
        Activation function to use for output layer.

    epochs : int, optional (default=500)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    G_layers : list, optional (default=[20,10,3,10,20])
        List that indicates the number of nodes per hidden layer for the
        generator. Thus, [10,10] indicates 2 hidden layers having each 10 nodes.

    D_layers : list, optional (default=[20,10,5])
        List that indicates the number of nodes per hidden layer for the
        discriminator. Thus, [10,10] indicates 2 hidden layers having each 10
        nodes.


    learning_rate: float in (0., 1), optional (default=0.001)
        learning rate of training the network

    index_D_layer_for_recon_error: int, optional (default = 1)
        This is the index of the hidden layer in the discriminator for which
        the reconstruction error will be determined between query sample and
        the sample created from the latent space.

    learning_rate_query: float in (0., 1), optional (default=0.001)
        learning rate for the backpropagation steps needed to find a point in
        the latent space of the generator that approximate the query sample


    epochs_query: int, optional (default=20) 
        Number of epochs to approximate the query sample in the latent space
        of the generator

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        Verbosity mode.
        - 0 = silent
        - 1 = progress bar

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data [0,1].
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

    def __init__(self, activation_hidden='tanh', dropout_rate=0.2,
                 latent_dim_G=2,
                 G_layers=[20, 10, 3, 10, 20], verbose=0, D_layers=[20, 10, 5],
                 index_D_layer_for_recon_error=1, epochs=500,
                 preprocessing=False,
                 learning_rate=0.001, learning_rate_query=0.01,
                 epochs_query=20,
                 batch_size=32, output_activation=None, contamination=0.1,
                 device=None):
        super(AnoGAN, self).__init__(contamination=contamination)

        self.activation_hidden = activation_hidden
        self.dropout_rate = dropout_rate
        self.latent_dim_G = latent_dim_G
        self.G_layers = G_layers
        self.D_layers = D_layers
        self.index_D_layer_for_recon_error = index_D_layer_for_recon_error
        self.output_activation = output_activation
        self.contamination = contamination
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_query = learning_rate_query
        self.epochs_query = epochs_query
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.verbose = verbose

        self.hist_loss_generator = []
        self.hist_loss_discriminator = []

        self.device = device

        check_parameter(dropout_rate, 0, 1,
                        param_name='dropout_rate', include_left=True)

    def plot_learning_curves(self, start_ind=0,
                             window_smoothening=10):  # pragma: no cover
        fig = plt.figure(figsize=(12, 5))
        l_gen = pd.Series(self.hist_loss_generator[start_ind:]).rolling(
            window_smoothening).mean()
        l_disc = pd.Series(self.hist_loss_discriminator[start_ind:]).rolling(
            window_smoothening).mean()

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(range(len(l_gen)), l_gen)
        ax.set_title('Generator')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Iter')

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(range(len(l_disc)), l_disc)
        ax.set_title('Discriminator')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Iter')

        plt.show()

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

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
        # train the discriminator and generator
        self.generator = Generator(latent_dim_G=self.latent_dim_G,
                                   n_features=self.n_features_,
                                   G_layers=self.G_layers,
                                   dropout_rate=self.dropout_rate,
                                   activation_hidden=self.activation_hidden,
                                   output_activation=self.output_activation)
        self.discriminator = Discriminator(n_features=self.n_features_,
                                           D_layers=self.D_layers,
                                           dropout_rate=self.dropout_rate,
                                           activation_hidden=self.activation_hidden)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate)
        optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X_norm)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        for n in range(self.epochs):
            if n % 100 == 0 and n != 0 and self.verbose == 1:
                print(f'Train iter: {n}')

            self.generator.train()
            self.discriminator.train()
            for X_train_ in dataloader:
                X_train_sel = X_train_[0].to(self.device)
                latent_noise = torch.rand(X_train_sel.size(
                    0), self.latent_dim_G, dtype=torch.float32).to(self.device)

                generated_data = self.generator(latent_noise)
                real_output = self.discriminator(X_train_sel)
                fake_output = self.discriminator(generated_data.detach())

                loss_D_real = nn.BCELoss()(real_output, torch.ones_like(
                    real_output) * 0.9).to(self.device)
                loss_D_fake = nn.BCELoss()(fake_output,
                                           torch.zeros_like(fake_output)).to(
                    self.device)
                loss_D = loss_D_real + loss_D_fake
                optimizer_d.zero_grad()
                loss_D.backward()
                optimizer_d.step()

                fake_output = self.discriminator(generated_data)
                loss_G = nn.BCELoss()(fake_output,
                                      torch.ones_like(fake_output)).to(
                    self.device)
                optimizer_g.zero_grad()
                loss_G.backward()
                optimizer_g.step()

                self.hist_loss_discriminator.append(loss_D.item())
                self.hist_loss_generator.append(loss_G.item())

        # Instantiate and train the query model
        self.generator.eval()
        self.discriminator.eval()
        self.query_model = QueryModel(
            self.generator, self.discriminator, self.latent_dim_G).to(
            self.device)
        optimizer_query = optim.Adam(
            self.query_model.parameters(), lr=self.learning_rate_query)
        scores = []
        # For each sample, use a few backpropagation steps to obtain a point in the latent space that best resembles the query sample
        self.query_model.train()
        for i in range(X_norm.shape[0]):
            if self.verbose == 1:
                print('query sample {} / {}'.format(i + 1, X_norm.shape[0]))

            query_sample = X_norm[[i],].to(self.device)
            assert (query_sample.shape[0] == 1)
            assert (query_sample.shape[1] == self.n_features_)

            # Make pseudo input (just zeros)
            zeros = torch.zeros((1, self.latent_dim_G)).to(self.device)

            # build model for back-propagating a approximate latent space where
            # reconstruction with query sample is optimal
            for i in range(self.epochs_query):
                if i % 25 == 0 and self.verbose == 1:
                    print('iter:', i)

                z, sample_gen, sample_disc_latent = self.query_model(zeros)
                with torch.no_grad():
                    sample_disc_latent_original = self.discriminator(
                        query_sample)
                # Reconstruction loss generator
                loss_recon_gen = torch.mean(torch.mean(
                    torch.abs(query_sample - sample_gen), axis=-1))
                # Reconstruction loss latent space of discrimator
                loss_recon_disc = torch.mean(torch.mean(
                    torch.abs(
                        sample_disc_latent_original - sample_disc_latent),
                    axis=-1))
                total_loss = loss_recon_gen + loss_recon_disc

                optimizer_query.zero_grad()
                total_loss.backward()
                optimizer_query.step()
            # Predict on X itself and calculate the reconstruction error as
            # the outlier scores.
            scores.append(total_loss.item())

        self.decision_scores_ = np.array(scores)
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
        check_is_fitted(self, ['decision_scores_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        X_norm = torch.tensor(X_norm, dtype=torch.float32)

        # Predict on X
        pred_scores = []

        self.query_model.eval()
        with torch.no_grad():
            for i in range(X_norm.shape[0]):
                if self.verbose == 1:
                    print(
                        'query sample {} / {}'.format(i + 1, X_norm.shape[0]))

                query_sample = X_norm[[i],].to(self.device)

                zeros = torch.zeros((1, self.latent_dim_G)).to(self.device)
                z, sample_gen, sample_disc_latent = self.query_model(zeros)
                sample_disc_latent_original = self.discriminator(query_sample)

                loss_recon_gen = torch.mean(torch.mean(
                    torch.abs(query_sample - sample_gen), axis=-1))
                loss_recon_disc = torch.mean(torch.mean(
                    torch.abs(
                        sample_disc_latent_original - sample_disc_latent),
                    axis=-1))
                total_loss = loss_recon_gen + loss_recon_disc
                pred_scores.append(total_loss.item())

        return np.array(pred_scores)
