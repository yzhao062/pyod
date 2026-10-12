# -*- coding: utf-8 -*-

"""Deep anomaly detection with deviation networks
Part of the codes are adapted from
https://github.com/GuansongPang/deviation-network
"""
# Author: Sihan Chen <schen976@usc.edu>
# License: BSD 2 clause


# Import necessary libraries
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import check_array
from torch.utils.data import Dataset, DataLoader

from .base import BaseDetector
from ..utils.torch_utility import TorchDataset

MAX_INT = np.iinfo(np.int32).max


# Define the network architectures
class DevNetD(nn.Module):
    def __init__(self, input_shape):
        super(DevNetD, self).__init__()
        self.fc1 = nn.Linear(input_shape, 1000)
        self.fc2 = nn.Linear(1000, 250)
        self.fc3 = nn.Linear(250, 20)
        self.score = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.score(x)
        return x


class DevNetS(nn.Module):
    def __init__(self, input_shape):
        super(DevNetS, self).__init__()
        self.fc1 = nn.Linear(input_shape, 1000)
        self.score = nn.Linear(1000, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.score(x)
        return x


class DevNetLinear(nn.Module):
    def __init__(self, input_shape):
        super(DevNetLinear, self).__init__()
        self.score = nn.Linear(input_shape, 1)

    def forward(self, x):
        x = self.score(x)
        return x


def deviation_loss(y_true, y_pred):
    '''
    Z-score-based deviation loss translated to PyTorch.
    '''
    confidence_margin = 5.0
    # size=5000 is the setting of l in algorithm 1 in the paper
    ref = torch.randn(5000, device=y_pred.device,
                      dtype=torch.float32)  # Generate normal distributed ref values
    dev = (y_pred - ref.mean()) / ref.std()
    inlier_loss = torch.abs(dev)
    outlier_loss = torch.abs(torch.clamp(confidence_margin - dev, min=0))

    # Compute the mean of the weighted sum of inlier and outlier losses
    return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


# Define the training and testing process
def train_and_test(model, train_loader, test_loader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
        print('Test Loss:', total_loss / len(test_loader))


# Main function to run the model
def deviation_network(input_shape, network_depth):
    '''
    Construct the deviation network-based detection model in PyTorch style
    '''
    # Select the model based on the network depth
    if network_depth == 4:
        model = DevNetD(input_shape)
    elif network_depth == 2:
        model = DevNetS(input_shape)
    elif network_depth == 1:
        model = DevNetLinear(input_shape)
    else:
        raise ValueError(
            "The network depth is not set properly")  # Use exception instead of sys.exit

    # Initialize the optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=0.001,
                              weight_decay=1e-6)  # Set clipnorm equivalent in PyTorch
    return model, optimizer


class SupDataset(Dataset):
    def __init__(self, x, outlier_indices, inlier_indices, rng):
        self.x = x
        self.outlier_indices = outlier_indices
        self.inlier_indices = inlier_indices
        self.rng = rng

    def __len__(self):
        return len(self.outlier_indices) + len(
            self.inlier_indices)  # or any other appropriate length

    def __getitem__(self, idx):
        if idx < len(self.inlier_indices):
            # Assuming inliers are processed first
            label = 0  # Assuming inlier label
            index = self.inlier_indices[idx]
        else:
            # Processing outliers
            label = 1  # Assuming outlier label
            index = self.outlier_indices[idx - len(self.inlier_indices)]

        return self.x[index], label


def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices,
                                      batch_size, rng):
    '''
    Batch generation for samples, alternating between positive and negative.
    Adjusted for use with PyTorch, handling data in tensors.
    '''
    training_data = []
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)

    for i in range(batch_size):
        if i % 2 == 0:
            sid = rng.choice(n_inliers, 1)
            training_data.append(x_train[inlier_indices[sid.item()]])
            training_labels.append(0)
        else:
            sid = rng.choice(n_outliers, 1)
            training_data.append(x_train[outlier_indices[sid.item()]])
            training_labels.append(1)

    # Convert lists to tensors
    training_data = torch.stack(training_data)
    training_labels = torch.tensor(training_labels, dtype=torch.long)

    return training_data, training_labels


def load_model_weight_predict(model, x_test, data_format=0):
    # Ensure x_test is a PyTorch tensor and also ensure it's on the same device as the model
    x_test = torch.tensor(x_test, dtype=torch.float32)

    if data_format == 0:
        scores = model(x_test)
    else:
        data_size = x_test.shape[0]
        scores = torch.zeros([data_size, ])
        batch_size = 512
        for i in range(0, data_size, batch_size):
            end = min(i + batch_size, data_size)
            subset = x_test[i:end]
            scores[i:end] = model(subset).flatten()

    # Make sure the output is flattened before returning
    scores = scores.flatten()  # Flatten the tensor to ensure it's one-dimensional

    return scores.detach().cpu().numpy()  # Convert to numpy array if needed


class DevNet(BaseDetector):
    """Deep anomaly detection with deviation networks (DevNet).

    DevNet is a weakly-supervised anomaly detector that trains a deep neural
    network using a small set of labeled anomalies alongside a larger body of
    unlabeled data.  A Z-score-based deviation loss encourages anomaly scores
    for the labeled outliers to deviate significantly above those of the
    reference (inlier) distribution.

    See :cite:`pang2019deep` for details.

    Parameters
    ----------
    network_depth : int, optional (default=2)
        Architecture selector.  ``1`` → linear model, ``2`` → one hidden
        layer (DevNetS), ``4`` → three hidden layers (DevNetD).

    batch_size : int, optional (default=512)
        Mini-batch size used for both training and inference.

    epochs : int, optional (default=50)
        Number of full passes over the training batches.

    nb_batch : ignored
        Deprecated and unused.  Accepted for backwards compatibility only;
        passing a non-None value emits a ``DeprecationWarning``.

    known_outliers : int, optional (default=30)
        Maximum number of labeled anomalies to retain when building the
        training set.  When the dataset contains more labeled anomalies, a
        random subset of this size is drawn (seeded by ``random_seed``).
        This mirrors the weakly-supervised setting described in the paper.

    cont_rate : ignored
        Deprecated and unused.  Accepted for backwards compatibility only;
        passing a non-None value emits a ``DeprecationWarning``.

    data_format : int, optional (default=0)
        Inference batching strategy.  ``0`` → score all samples in one
        forward pass; ``1`` → iterate in chunks of 512, which is useful
        when the test set is very large.

    random_seed : int or None, optional (default=42)
        Seed for ``random``, ``numpy.random``, and ``torch`` so that
        training is reproducible across calls to :meth:`fit`.
        Pass ``None`` to skip deterministic seeding.

    device : str or torch.device, optional (default=None)
        Torch device.  When ``None`` the first available CUDA device is
        used, falling back to CPU.

    contamination : float in (0., 0.5), optional (default=0.1)
        Proportion of outliers in the dataset, used to set the decision
        threshold after fitting.

    Attributes
    ----------
    model : torch.nn.Module
        Fitted PyTorch network.

    decision_scores_ : numpy array of shape (n_samples,)
        Outlier scores computed on the training data.  Higher values are
        more anomalous.

    threshold_ : float
        Cut-off score that separates inliers from outliers.

    labels_ : numpy array of shape (n_samples,)
        Binary labels assigned to the training data after fitting
        (``0`` = inlier, ``1`` = outlier).
    """

    def __init__(self,
                 network_depth=2,
                 batch_size=512,
                 epochs=50,
                 nb_batch=None,
                 known_outliers=30,
                 cont_rate=None,
                 data_format=0,
                 random_seed=42,
                 device=None,
                 contamination=0.1):
        super(DevNet, self).__init__(contamination=contamination)
        self._classes = 2
        self.network_depth = network_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.known_outliers = known_outliers
        self.data_format = data_format
        self.random_seed = random_seed
        self.device = device
        if nb_batch is not None:
            warnings.warn(
                "nb_batch is not used by DevNet and will be removed in a "
                "future release.", DeprecationWarning, stacklevel=2)
        self.nb_batch = nb_batch
        if cont_rate is not None:
            warnings.warn(
                "cont_rate is not used by DevNet and will be removed in a "
                "future release.", DeprecationWarning, stacklevel=2)
        self.cont_rate = cont_rate
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        rng = np.random.RandomState(self.random_seed)

        outlier_indices = np.where(y == 1)[0]
        inlier_indices = np.where(y == 0)[0]
        n_outliers = len(outlier_indices)
        print("Original training size: %d, No. outliers: %d" % (
            X.shape[0], n_outliers))

        # Cap the labeled-anomaly set to self.known_outliers, as in the paper.
        if n_outliers > self.known_outliers:
            outlier_indices = rng.choice(
                outlier_indices, self.known_outliers, replace=False)

        print(y.shape[0], outlier_indices.shape[0], inlier_indices.shape[0])

        self.model, optimizer = deviation_network(X.shape[1],
                                                  self.network_depth)
        train_dataset = SupDataset(X, outlier_indices, inlier_indices, rng)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True)

        def train_model(model, data_loader, epochs):
            model.train()
            for epoch in range(epochs):
                for data, labels in data_loader:
                    data, labels = data.to(torch.float32), labels.to(
                        torch.float32)  # Ensure data types
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = deviation_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        # Training the model
        train_model(self.model, train_loader, epochs=self.epochs)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        X = check_array(X)

        dataset = TorchDataset(X=X, return_idx=True)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                # this is the outlier score
                outlier_scores[data_idx] = load_model_weight_predict(
                    self.model, data, self.data_format)
        return outlier_scores

    def fit_predict_score(self, X, y, scoring='roc_auc_score'):
        """
        Fit the detector with labels, predict on samples, and evaluate the model by predefined metrics.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : numpy array of shape (n_samples,)
            The labels or target values corresponding to X.
        scoring : str, optional (default='roc_auc_score')
            Evaluation metric:
            - 'roc_auc_score': ROC score
            - 'prc_n_score': Precision @ rank n score

        Returns
        -------
        score : float
        """

        # Fit the model with both X and y
        self.fit(X, y)

        # Prediction and scoring
        if scoring == 'roc_auc_score':
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y, self.decision_scores_)
        elif scoring == 'prc_n_score':
            from sklearn.metrics import precision_recall_curve
            precision, _, _ = precision_recall_curve(y, self.decision_scores_)
            score = precision[
                1]  # Assuming this is how you'd compute Precision @ rank n
        else:
            raise NotImplementedError('PyOD built-in scoring only supports '
                                      'ROC and Precision @ rank n')

        print("{metric}: {score}".format(metric=scoring, score=score))

        return score
