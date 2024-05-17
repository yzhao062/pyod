# -*- coding: utf-8 -*-
"""Multiple-Objective Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""
# Author: Winston Li <jk_zhengli@hotmail.com>
# License: BSD 2 clause

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

from .base import BaseDetector
from .gaal_base_torch import create_discriminator, create_generator


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)

        return torch.from_numpy(sample), idx




class MO_GAAL(BaseDetector):
    """Multi-Objective Generative Adversarial Active Learning.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set.

    k : int, optional (default=10)
        The number of sub generators.

    stop_epochs : int, optional (default=20)
        The number of epochs of training.

    lr_d : float, optional (default=0.01)
        The learning rate of the discriminator.

    lr_g : float, optional (default=0.0001)
        The learning rate of the generator.

    momentum : float, optional (default=0.9)
        The momentum parameter for the optimizer.
    """

    def __init__(self, k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001, momentum=0.9, contamination=0.1):
        super(MO_GAAL, self).__init__(contamination=contamination)
        self.k = k
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.momentum = momentum

    def fit(self, X):
        torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.discriminator = create_discriminator(X.size(1), X.size(0)).to(device)
        self.generators = [create_generator(X.size(1)).to(device) for _ in range(self.k)]

        optimizer_d = torch.optim.SGD(self.discriminator.parameters(), lr=self.lr_d, momentum=self.momentum)
        optimizers_g = [torch.optim.SGD(gen.parameters(), lr=self.lr_g, momentum=self.momentum) for gen in
                        self.generators]

        criterion = torch.nn.BCELoss()

        epochs = self.stop_epochs * 3
        batch_size = min(500, X.size(0))
        num_batches = X.size(0) // batch_size

        decision_scores = []  # List to gather all scores (assuming binary classification for simplicity)
        for epoch in range(epochs):
            for index in range(num_batches):
                real_data = X[index * batch_size:(index + 1) * batch_size]
                real_labels = torch.ones(real_data.size(0), 1).to(device)
                fake_labels = torch.zeros(real_data.size(0), 1).to(device)

                # Train discriminator
                self.discriminator.zero_grad()
                outputs_real = self.discriminator(real_data)
                loss_real = criterion(outputs_real, real_labels)
                loss_real.backward()

                # Assuming discriminator output as decision scores
                decision_scores.extend(outputs_real.detach().cpu().numpy())

                for gen, optimizer_g in zip(self.generators, optimizers_g):
                    gen.zero_grad()
                    noise = torch.randn(real_data.size(0), X.size(1)).to(device)
                    fake_data = gen(noise)
                    outputs_fake = self.discriminator(fake_data)
                    loss_fake = criterion(outputs_fake, fake_labels)
                    loss_fake.backward()
                    optimizer_g.step()

                optimizer_d.step()

        # Finalize decision scores and labels
        self.decision_scores_ = np.array([self.decision_function(X[i:i + 1]) for i in range(len(X))]).flatten()
        self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        """Calculate the anomaly scores for input samples X.

        Parameters
        ----------
        X : torch.Tensor
            Input samples.

        Returns
        -------
        scores : torch.Tensor
            Anomaly scores for each sample.
        """
        # Ensuring the discriminator has been moved to a device
        if next(self.discriminator.parameters()).is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        X = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = self.discriminator(X).squeeze()
        return scores.cpu().numpy()

# Further methods and attributes can be added as needed to match the full functionality.
