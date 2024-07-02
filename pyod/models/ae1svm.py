# -*- coding: utf-8 -*-
"""Using AE-1SVM with Outlier Detection (PyTorch)
    Source: https://arxiv.org/pdf/1804.04888
"""
# Author: Zhuo Xiao <zhuoxiao@usc.edu>

from __future__ import division, print_function

import numpy as np
import torch
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn

from .base import BaseDetector
from ..utils.stat_models import pairwise_distances_no_broadcast
from ..utils.torch_utility import get_activation_by_name


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader"""

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx, :]
        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
        return torch.from_numpy(sample), idx


class InnerAE1SVM(nn.Module):
    def __init__(self, n_features, encoding_dim, rff_dim, sigma=1.0, hidden_neurons=(128, 64),
                 dropout_rate=0.2, batch_norm=True, hidden_activation='relu'):
        super(InnerAE1SVM, self).__init__()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.rff = RandomFourierFeatures(encoding_dim, rff_dim, sigma)
        self.svm_weights = nn.Parameter(torch.randn(rff_dim))
        self.svm_bias = nn.Parameter(torch.randn(1))

        activation = get_activation_by_name(hidden_activation)
        layers_neurons_encoder = [n_features, *hidden_neurons, encoding_dim]

        for idx in range(len(layers_neurons_encoder) - 1):
            self.encoder.add_module(f"linear{idx}",
                                    nn.Linear(layers_neurons_encoder[idx], layers_neurons_encoder[idx + 1]))
            if batch_norm:
                self.encoder.add_module(f"batch_norm{idx}", nn.BatchNorm1d(layers_neurons_encoder[idx + 1]))
            self.encoder.add_module(f"activation{idx}", activation)
            self.encoder.add_module(f"dropout{idx}", nn.Dropout(dropout_rate))

        layers_neurons_decoder = layers_neurons_encoder[::-1]

        for idx in range(len(layers_neurons_decoder) - 1):
            self.decoder.add_module(f"linear{idx}",
                                    nn.Linear(layers_neurons_decoder[idx], layers_neurons_decoder[idx + 1]))
            if batch_norm and idx < len(layers_neurons_decoder) - 2:
                self.decoder.add_module(f"batch_norm{idx}", nn.BatchNorm1d(layers_neurons_decoder[idx + 1]))
            self.decoder.add_module(f"activation{idx}", activation)
            if idx < len(layers_neurons_decoder) - 2:
                self.decoder.add_module(f"dropout{idx}", nn.Dropout(dropout_rate))

    def forward(self, x):
        x = self.encoder(x)
        rff_features = self.rff(x)
        x = self.decoder(x)
        return x, rff_features

    def svm_decision_function(self, rff_features):
        return torch.matmul(rff_features, self.svm_weights) + self.svm_bias


class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=1.0):
        super(RandomFourierFeatures, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * sigma)
        self.bias = nn.Parameter(torch.randn(output_dim) * 2 * np.pi)

    def forward(self, x):
        x = torch.matmul(x, self.weights) + self.bias
        return torch.cos(x)


class AE1SVM(BaseDetector):
    """Auto Encoder with One-class SVM for anomaly detection."""

    def __init__(self, hidden_neurons=None, hidden_activation='relu',
                 batch_norm=True, learning_rate=1e-3, epochs=50, batch_size=32,
                 dropout_rate=0.2, weight_decay=1e-5, preprocessing=True,
                 loss_fn=None, contamination=0.1, alpha=1.0, sigma=1.0, nu=0.1, kernel_approx_features=1000):
        super(AE1SVM, self).__init__(contamination=contamination)

        self.model = None
        self.decision_scores_ = None
        self.std = None
        self.mean = None
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.alpha = alpha
        self.sigma = sigma
        self.nu = nu
        self.kernel_approx_features = kernel_approx_features

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)

        n_samples, n_features = X.shape
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            self.std[self.std == 0] = 1e-6  # Avoid division by zero
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            train_set = PyODDataset(X=X)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.model = InnerAE1SVM(n_features=n_features, encoding_dim=32, rff_dim=self.kernel_approx_features,
                                 sigma=self.sigma,
                                 hidden_neurons=self.hidden_neurons, dropout_rate=self.dropout_rate,
                                 batch_norm=self.batch_norm, hidden_activation=self.hidden_activation)
        self.model = self.model.to(self.device)
        self._train_autoencoder(train_loader)

        if self.best_model_dict is not None:
            self.model.load_state_dict(self.best_model_dict)
        else:
            raise ValueError('Training failed, no valid model state found')

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def _train_autoencoder(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(self.epochs):
            overall_loss = []
            for data, data_idx in train_loader:
                data = data.to(self.device).float()
                reconstructions, rff_features = self.model(data)
                recon_loss = self.loss_fn(data, reconstructions)
                svm_scores = self.model.svm_decision_function(rff_features)
                svm_loss = torch.mean(torch.clamp(1 - svm_scores, min=0))

                loss = self.alpha * recon_loss + svm_loss
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {np.mean(overall_loss)}')

            if np.mean(overall_loss) < self.best_loss:
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = self.model.state_dict()

    def decision_function(self, X):
        check_is_fitted(self, ['model', 'best_model_dict'])
        X = check_array(X)
        dataset = PyODDataset(X=X, mean=self.mean, std=self.std) if self.preprocessing else PyODDataset(X=X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        outlier_scores = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data = data.to(self.device).float()
                reconstructions, rff_features = self.model(data)
                scores = pairwise_distances_no_broadcast(data.cpu().numpy(), reconstructions.cpu().numpy())
                outlier_scores[data_idx] = scores
        return outlier_scores
