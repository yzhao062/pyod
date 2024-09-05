# -*- coding: utf-8 -*-
"""LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks
"""
# Author: Adam Goodge <a.goodge@u.nus.edu>
#

from copy import deepcopy

import numpy as np

try:
    import torch
except ImportError:
    print('please install torch first')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


# negative samples for training
def generate_negative_samples(x, sample_type, proportion, epsilon):
    n_samples = int(proportion * (len(x)))
    n_dim = x.shape[-1]

    # uniform samples in range [x.min(),x.max()]
    rand_unif = x.min() + (x.max() - x.min()) * np.random.rand(n_samples,
                                                               n_dim).astype(
        'float32')
    # subspace perturbation samples
    x_temp = x[np.random.choice(np.arange(len(x)), size=n_samples)]
    randmat = np.random.rand(n_samples, n_dim) < 0.3
    rand_sub = x_temp + randmat * (
            epsilon * np.random.randn(n_samples, n_dim)).astype('float32')

    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    if sample_type == 'SUBSPACE':
        neg_x = rand_sub
    if sample_type == 'MIXED':
        # randomly sample from uniform and gaussian negative samples
        neg_x = np.concatenate((rand_unif, rand_sub), 0)
        neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size=n_samples)]

    neg_y = np.ones(len(neg_x))

    return neg_x.astype('float32'), neg_y.astype('float32')


class SCORE_MODEL(nn.Module):
    def __init__(self, k):
        super(SCORE_MODEL, self).__init__()
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(k, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.network(x)
        out = torch.squeeze(out, 1)
        return out


class WEIGHT_MODEL(nn.Module):
    def __init__(self, k):
        super(WEIGHT_MODEL, self).__init__()
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(k, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, k),
        )
        self.final_norm = nn.BatchNorm1d(1)

    def forward(self, x):
        alpha = self.network(x)
        # get weights > 0 and sum to 1.0
        alpha = F.softmax(alpha, dim=1)
        # multiply weights by each distance in input vector
        out = torch.sum(alpha * x, dim=1, keepdim=True)
        # batch norm
        out = self.final_norm(out)
        out = torch.squeeze(out, 1)
        return out


class LUNAR(BaseDetector):
    """
    LUNAR class for outlier detection. See https://www.aaai.org/AAAI22Papers/AAAI-51.GoodgeA.pdf for details.
    For an observation, its ordered list of distances to its k nearest neighbours is input to a neural network, 
    with one of the following outputs:

        1) SCORE_MODEL: network directly outputs the anomaly score.
        2) WEIGHT_MODEL: network outputs a set of weights for the k distances, the anomaly score is then the
            sum of weighted distances.

    See :cite:`goodge2022lunar` for details.

    Parameters
    ----------
    model_type: str in ['WEIGHT', 'SCORE'], optional (default = 'WEIGHT')
        Whether to use WEIGHT_MODEL or SCORE_MODEL for anomaly scoring.

    n_neighbors: int, optional (default = 5)
        Number of neighbors to use by default for k neighbors queries.

    negative_sampling: str in ['UNIFORM', 'SUBSPACE', MIXED'], optional (default = 'MIXED)
        Type of negative samples to use between:

        - 'UNIFORM': uniformly distributed samples
        - 'SUBSPACE': subspace perturbation (additive random noise in a subset of features)
        - 'MIXED': a combination of both types of samples

    val_size: float in [0,1], optional (default = 0.1)
        Proportion of samples to be used for model validation

    scaler: object in {StandardScaler(), MinMaxScaler(), optional (default = MinMaxScaler())
        Method of data normalization

    epsilon: float, optional (default = 0.1)
        Hyper-parameter for the generation of negative samples. 
        A smaller epsilon results in negative samples more similar to normal samples.

    proportion: float, optional (default = 1.0)
        Hyper-parameter for the proprotion of negative samples to use relative to the 
        number of normal training samples.

    n_epochs: int, optional (default = 200)
        Number of epochs to train neural network.

    lr: float, optional (default = 0.001)
        Learning rate.

    wd: float, optional (default = 0.1)
        Weight decay.
    
    verbose: int in {0,1}, optional (default = 0):
        To view or hide training progress

    Attributes
    ----------
    """

    def __init__(self, model_type="WEIGHT", n_neighbours=5,
                 negative_sampling="MIXED",
                 val_size=0.1, scaler=MinMaxScaler(), epsilon=0.1,
                 proportion=1.0,
                 n_epochs=200, lr=0.001, wd=0.1, verbose=0, contamination=0.1):
        super(LUNAR, self).__init__(contamination=contamination)

        self.model_type = model_type
        self.n_neighbours = n_neighbours
        self.negative_sampling = negative_sampling
        self.epsilon = epsilon
        self.proportion = proportion
        self.n_epochs = n_epochs
        self.scaler = scaler
        self.lr = lr
        self.wd = wd
        self.val_size = val_size
        self.verbose = verbose
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == "SCORE":
            self.network = SCORE_MODEL(n_neighbours).to(self.device)
        elif model_type == "WEIGHT":
            self.network = WEIGHT_MODEL(n_neighbours).to(self.device)

    def fit(self, X, y=None):
        """Fit detector. y is assumed to be 0 for all training samples.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Overwritten with 0 for all training samples (assumed to be normal).
        Returns
        -------
        self : object
            Fitted estimator.
        """

        # X = check_array(X)
        self._set_n_classes(y)
        X = X.astype('float32')
        y = np.zeros(len(X))

        # split training and validation sets
        train_x, val_x, train_y, val_y = train_test_split(X, y,
                                                          test_size=self.val_size)

        # fit data scaler to the training set if scaler has been passed
        if (self.scaler == None):
            pass
        else:
            self.scaler.fit(train_x)

        # scale data if scaler has been passed
        if (self.scaler == None):
            pass
        else:
            train_x = self.scaler.transform(train_x)
            val_x = self.scaler.transform(val_x)

        # generate negative samples for training and validation set seperately 
        neg_train_x, neg_train_y = generate_negative_samples(train_x,
                                                             self.negative_sampling,
                                                             self.proportion,
                                                             self.epsilon)
        neg_val_x, neg_val_y = generate_negative_samples(val_x,
                                                         self.negative_sampling,
                                                         self.proportion,
                                                         self.epsilon)

        train_x = np.vstack((train_x, neg_train_x))
        train_y = np.hstack((train_y, neg_train_y))
        val_x = np.vstack((val_x, neg_val_x))
        val_y = np.hstack((val_y, neg_val_y))

        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbours + 1)
        self.neigh.fit(train_x)

        # nearest neighbours of training set
        train_dist, _ = self.neigh.kneighbors(train_x[train_y == 0],
                                              n_neighbors=self.n_neighbours + 1)
        neg_train_dist, _ = self.neigh.kneighbors(train_x[train_y == 1],
                                                  n_neighbors=self.n_neighbours)
        # remove self loops of normal training points
        train_dist = np.vstack((train_dist[:, 1:], neg_train_dist))
        # nearest neighbours of validation set
        val_dist, _ = self.neigh.kneighbors(val_x,
                                            n_neighbors=self.n_neighbours)

        train_dist = torch.tensor(train_dist, dtype=torch.float32).to(
            self.device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        val_dist = torch.tensor(val_dist, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(val_y, dtype=torch.float32).to(self.device)
        # loss function
        criterion = nn.MSELoss(reduction='none')
        # optimizer
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        # for early stopping
        best_val_score = 0
        # model training
        for epoch in range(self.n_epochs):

            # see performance of model before epoch
            with torch.no_grad():

                self.network.eval()

                out = self.network(train_dist)
                train_score = roc_auc_score(train_y.cpu(), out.cpu())

                out = self.network(val_dist)
                val_score = roc_auc_score(val_y.cpu(), out.cpu())

                # save best model
                if val_score >= best_val_score:
                    best_dict = {'epoch': epoch,
                                 'model_state_dict': deepcopy(
                                     self.network.state_dict()),
                                 'optimizer_state_dict': deepcopy(
                                     optimizer.state_dict()),
                                 'train_score': train_score,
                                 'val_score': val_score,
                                 }

                    # reset current best score
                    best_val_score = val_score

                if self.verbose == 1:
                    print(
                        f"Epoch {epoch} \t Train Score {np.round(train_score, 6)} \t Val Score {np.round(val_score, 6)}")

            # training loop
            self.network.train()
            optimizer.zero_grad()
            out = self.network(train_dist)
            loss = criterion(out, train_y).sum()
            loss.backward()
            optimizer.step()

        # print best model after training
        if self.verbose == 1:
            print(
                f"Finished training...\nBest Model: Epoch {best_dict['epoch']} \t Train Score {best_dict['train_score']} \t Val Score {best_dict['val_score']}")
        # load best model after training
        self.network.load_state_dict(best_dict['model_state_dict'])

        # Determine outlier scores for train set
        # scale data if scaler has been passed
        if (self.scaler == None):
            X_norm = np.copy(X)
        else:
            X_norm = self.scaler.transform(X)

        # nearest neighbour search
        dist, _ = self.neigh.kneighbors(X_norm, self.n_neighbours)
        dist = torch.tensor(dist, dtype=torch.float32).to(self.device)
        # forward pass
        with torch.no_grad():
            self.network.eval()
            anomaly_scores = self.network(dist)

        self.decision_scores_ = anomaly_scores.cpu().detach().numpy().ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        For consistency, outliers are assigned with larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        check_is_fitted(self, ['decision_scores_'])
        # X = check_array(X)
        X = X.astype('float32')

        # scale data
        if (self.scaler == None):
            pass
        else:
            X = self.scaler.transform(X)

        # nearest neighbour search
        dist, _ = self.neigh.kneighbors(X, self.n_neighbours)
        dist = torch.tensor(dist, dtype=torch.float32).to(self.device)
        # forward pass
        with torch.no_grad():
            self.network.eval()
            anomaly_scores = self.network(dist)

        scores = anomaly_scores.cpu().detach().numpy().ravel()

        return scores
