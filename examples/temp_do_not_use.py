# -*- coding: utf-8 -*-
"""Using Auto Encoder with Outlier Detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyod.utils.utility import check_parameter
from pyod.utils.stat_models import pairwise_distances_no_broadcast

from pyod.models.base import BaseDetector
from pyod.utils.data import generate_data
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from torch import nn
from torchvision import transforms


# %% Dataset to manage vector to vector data
class PyODDataset(torch.utils.data.Dataset):
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

        if self.mean.any():
            sample = (sample - mean) / std

        return torch.from_numpy(sample), idx


contamination = 0.1  # percentage of outliers
n_train = 20000  # number of training points
n_test = 2000  # number of testing points
n_features = 200  # number of features

# Generate sample data
X_train, y_train, X_test, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=n_features,
                  contamination=contamination,
                  random_state=42)

# create mean and std for the data
# this is light so could be done in numpy
mean, std = np.mean(X_train, axis=0), np.mean(X_train, axis=0)

train_set = PyODDataset(X=X_train, mean=mean, std=std)
test_set = PyODDataset(X=X_train, mean=mean, std=std)

num_epochs = 20
batch_size = 256
learning_rate = 1e-3

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=batch_size,
                                          shuffle=True)

# for idx, batch in enumerate(train_loader):
#     print(idx, batch.shape)
#     # print(batch[0, :])

for idx, batch in enumerate(train_loader):
    print(idx, batch[0].shape, batch[1].shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class inner_autoencoder(nn.Module):
    def __init__(self, hidden_neurons=[128, 64],
                 dropout_rate=0.2,
                 batch_norm=True,
                 preprocessing=True):
        super(inner_autoencoder, self).__init__()
        self.n_features_ = 200
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        self.layers_neurons_ = [self.n_features_, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        # print(self.layers_neurons_decoder_)

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if batch_norm:
                self.encoder.add_module("batch_norm" + str(idx),
                                        nn.BatchNorm1d(
                                            self.layers_neurons_[idx]))
            self.encoder.add_module("linear" + str(idx),
                                    torch.nn.Linear(self.layers_neurons_[idx],
                                                    self.layers_neurons_[
                                                        idx + 1]))
            self.encoder.add_module("relu" + str(idx), torch.nn.ReLU())
            self.encoder.add_module("dropout" + str(idx),
                                    torch.nn.Dropout(dropout_rate))

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if batch_norm:
                self.decoder.add_module("batch_norm" + str(idx),
                                        nn.BatchNorm1d(
                                            self.layers_neurons_decoder_[idx]))
            self.decoder.add_module("linear" + str(idx), torch.nn.Linear(
                self.layers_neurons_decoder_[idx],
                self.layers_neurons_decoder_[idx + 1]))
            self.decoder.add_module("relu" + str(idx), torch.nn.ReLU())
            self.decoder.add_module("dropout" + str(idx),
                                    torch.nn.Dropout(dropout_rate))

    def forward(self, x):
        # we could return the latent representation here after the encoder as the latent representation
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# in the fit function
# training process
model = inner_autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

best_loss = float('inf')
best_model_dict = None
for epoch in range(num_epochs):
    overall_loss = []
    for data, data_idx in train_loader:
        # print(epoch, data.shape)
        data = data.cuda().float()
        loss = criterion(data, model(data))
        # print('epoch {epoch} '.format(epoch=epoch), loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()
        overall_loss.append(loss.item())
        # print(loss.item())
    print('epoch {epoch} '.format(epoch=epoch), np.mean(overall_loss))

    # track the best model so far
    if np.mean(overall_loss) <= best_loss:
        print("epoch {ep} is the current best; loss={loss}".format(ep=epoch,
                                                                   loss=np.mean(
                                                                       overall_loss)))
        best_loss = np.mean(overall_loss)
        best_model_dict = model.state_dict()

# evaluation model
# get latent representation
best_model = inner_autoencoder().to(device)
best_model.load_state_dict(best_model_dict)
best_model(torch.from_numpy(X_train).float().cuda())

# %%
best_model.eval()

X_reconst = np.zeros([n_train, ])
with torch.no_grad():
    for idx, batch in enumerate(train_loader):
        # print(epoch, data.shape)
        data = batch[0].cuda().float()
        idx = batch[1]
        # this is the outlier score
        X_reconst[idx] = pairwise_distances_no_broadcast(batch[0], best_model(
            data).cpu().numpy())


# %%

class AutoEncoder(BaseDetector):
    def __init__(self,
                 hidden_neurons=None,
                 # hidden_activation='relu',
                 # output_activation='sigmoid',
                 batch_norm=True,
                 # loss='mse',
                 # optimizer='adam',
                 learning_rate=1e-3,
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 # l2_regularizer=0.1,
                 weight_decay=1e-5,
                 # validation_size=0.1,
                 preprocessing=True,
                 # verbose=1,
                 # random_state=None,
                 contamination=0.1,
                 device=None):
        super(AutoEncoder, self).__init__(contamination=contamination)
        self.hidden_neurons = hidden_neurons
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate

        self.epochs = epochs
        self.batch_size = batch_size

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [128, 64]

        # if optimizer == 'adam':
        #     self.optimizer = torch.optim.Adam(
        #         model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # print(self.optimizer)
        # self.hidden_activation = hidden_activation
        # self.output_activation = output_activation
        # self.loss = loss
        # fix to adam
        # self.optimizer = optimizer

        # self.l2_regularizer = l2_regularizer
        # self.validation_size = validation_size

        # self.verbose = verbose
        # self.random_state = random_state

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

        self.mean, self.std = np.mean(X_train, axis=0), np.mean(X_train,
                                                                axis=0)

        train_set = PyODDataset(X=X_train, mean=self.mean, std=self.std)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # initialize the model
        self.model = inner_autoencoder(
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            preprocessing=self.preprocessing)

        self.model = self.model.to(self.device)
        # print model information
        print(self.model)
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X_train)

    def _train_autoencoder(self, train_loader):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(self.epochs):
            overall_loss = []
            for data, data_idx in train_loader:
                # print(epoch, data.shape)
                data = data.to(self.device).float()
                # print(data.shape)
                loss = criterion(data, self.model(data))
                # print('epoch {epoch} '.format(epoch=epoch), loss.item())

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
                # print(loss.item())
            print('epoch {epoch}: training loss {train_loss} '.format(
                epoch=epoch, train_loss=np.mean(overall_loss)))

            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                # print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = self.model.state_dict()

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
        self.model.eval()

        dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)

        X_reconst = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for data, data_idx in train_loader:
                # print(epoch, data.shape)
                data_cuda = data.to(self.device).float()
                # idx = batch[1]
                # this is the outlier score
                X_reconst[data_idx] = pairwise_distances_no_broadcast(data,
                                                                      self.model(
                                                                          data_cuda).cpu().numpy())

        return X_reconst


clf = AutoEncoder(epochs=10)
clf.fit(X_train)
