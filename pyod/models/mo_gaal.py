# -*- coding: utf-8 -*-
"""Multiple-Objective Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import math
import torch.nn.functional as F

from .base import BaseDetector
from .gaal_base import create_discriminator, create_generator

class PyODDataset(torch.utils.data.Dataset):
    """Custom Dataset for handling data operations in PyTorch for outlier detection."""

    def __init__(self, X):
        super(PyODDataset, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class MO_GAAL(BaseDetector):
    """Multi-Objective Generative Adversarial Active Learning.

    MO_GAAL directly generates informative potential outliers to assist the
    classifier in describing a boundary that can separate outliers from normal
    data effectively. Moreover, to prevent the generator from falling into the
    mode collapsing problem, the network structure of SO-GAAL is expanded from
    a single generator (SO-GAAL) to multiple generators with different
    objectives (MO-GAAL) to generate a reasonable reference distribution for
    the whole dataset.
    Read more in the :cite:`liu2019generative`.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    k : int, optional (default=10)
        The number of sub generators.

    stop_epochs : int, optional (default=10)
        The training epoch at which training will be stopped.

    epochs : int, optional (default=100)
        The number of training epochs.

    lr_d : float, optional (default=0.001)
        The learning rate of the discriminator.

    lr_g : float, optional (default=0.001)
        The learning rate of the generator.

    decay : float, optional (default=0.5)
        The decay of the learning rate.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.

    labels_ : numpy array of shape (n_samples,)
        The binary labels of the training data.
        0 stands for inliers and 1 for outliers.

    """

    def __init__(self, contamination=0.1, k=10, stop_epochs=10, epochs=100,
                 lr_d=0.001, lr_g=0.001, decay=0.5):
        super(MO_GAAL, self).__init__(contamination=contamination)
        self.k = k
        self.stop_epochs = stop_epochs
        self.epochs = epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.decay = decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self._set_n_classes(y)
        latent_size = X.shape[1]
        data_size = X.shape[0]

        dataloader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32).to(self.device)), batch_size=min(500, len(X)), shuffle=True)

        self.discriminator = create_discriminator(latent_size, data_size).to(self.device)
        self.generators = [create_generator(latent_size).to(self.device) for _ in range(self.k)]

        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        g_optimizers = [optim.Adam(generator.parameters(), lr=self.lr_g) for generator in self.generators]

        criterion = nn.BCELoss()

        self.train_history = defaultdict(list)
        stop = 0
        for epoch in range(self.epochs):
            if stop == 1:
                break
            for batch_data in dataloader:
                batch_data = batch_data[0].to(self.device)
                real_labels = torch.ones(batch_data.size(0), 1).to(self.device)
                fake_labels = torch.zeros(batch_data.size(0), 1).to(self.device)

                # Train Discriminator
                self.discriminator.zero_grad()
                outputs = self.discriminator(batch_data)
                real_loss = criterion(outputs, real_labels)
                real_loss.backward()

                d_loss_total = real_loss.item()

                for i in range(self.k):
                    noise = torch.randn(batch_data.size(0), latent_size).to(self.device)
                    fake_data = self.generators[i](noise)
                    outputs = self.discriminator(fake_data.detach())
                    fake_loss = criterion(outputs, fake_labels)
                    fake_loss.backward()
                    d_loss_total += fake_loss.item()

                d_optimizer.step()

                # Train Generators
                for i in range(self.k):
                    self.generators[i].zero_grad()
                    noise = torch.randn(batch_data.size(0), latent_size).to(self.device)
                    fake_data = self.generators[i](noise)
                    outputs = self.discriminator(fake_data)
                    g_loss = criterion(outputs, real_labels)
                    g_loss.backward()
                    g_optimizers[i].step()

                    self.train_history[f'sub_generator{i}_loss'].append(g_loss.item())

                generator_loss = np.mean([self.train_history[f'sub_generator{i}_loss'][-1] for i in range(self.k)])
                self.train_history['generator_loss'].append(generator_loss)

                if epoch + 1 > self.stop_epochs:
                    stop = 1

        decision_scores = self.discriminator(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().detach().numpy()
        self.decision_scores_ = decision_scores.ravel()
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
        check_is_fitted(self, ['discriminator'])
        X = check_array(X)
        pred_scores = self.discriminator(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().detach().numpy().ravel()
        return pred_scores
