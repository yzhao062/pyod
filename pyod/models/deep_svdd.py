# -*- coding: utf-8 -*-
"""Deep One-Class Classification for outlier detection
"""
# Author: Rafal Bodziony <bodziony.rafal@gmail.com> for the TensorFlow version
# Author: Yuehan Qin <yuehanqi@usc.edu> for the PyTorch version
# License: BSD 2 clause


import numpy as np

try:
    import torch
except ImportError:
    print('please install torch first')

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDetector
from ..utils.torch_utility import get_activation_by_name
from ..utils.utility import check_parameter

optimizer_dict = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta,
    'adamw': optim.AdamW,
    'nadam': optim.NAdam,
    'sparseadam': optim.SparseAdam,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS
}


class InnerDeepSVDD(nn.Module):
    """Inner class for DeepSVDD model.

    Parameters
    ----------
    n_features:
        Number of features in the input data.

    use_ae: bool, optional (default=False)
        The AutoEncoder type of DeepSVDD it reverse neurons from hidden_neurons
        if set to True.

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. if use_ae is True, neurons
        will be reversed eg. [64, 32] -> [64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
    """

    def __init__(self, n_features, use_ae,
                 hidden_neurons, hidden_activation,
                 output_activation,
                 dropout_rate, l2_regularizer, feature_type, input_shape=None):
        super(InnerDeepSVDD, self).__init__()
        self.use_ae = use_ae
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.feature_type = feature_type
        self.input_shape = input_shape
        if self.feature_type == "obs":
            self.embedder_features = n_features
            self.linear_features = n_features
            self.embedder = self._build_embedder()
        elif self.feature_type in ["hidden", "dist"]:
            self.linear_features = self.input_shape[1]
        elif self.feature_type == "hidden_obs":
            self.embedder_features = n_features
            self.linear_features = n_features + self.input_shape[-1]
            self.embedder = self._build_embedder()
        self.fc_part = self._build_fc()
        self.c = None  # Center of the hypersphere for DeepSVDD

    def _init_c(self, X_norm, eps=0.1):
        intermediate_output = {}
        hook_handle = self.fc_part._modules.get('net_output').register_forward_hook(
            lambda module, input, output: intermediate_output.update({'net_output': output})
        )
        if self.feature_type in ["obs", "hidden", "dist"]:
            output = self.forward(X_norm)
        elif self.feature_type == "hidden_obs":
            output = self.forward([X_norm[0], X_norm[1]])
        out = intermediate_output['net_output']
        hook_handle.remove()
        self.c = torch.mean(out, dim=0)
        self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(torch.abs(self.c) < eps) & (self.c > 0)] = eps

    def _build_embedder(self):
        if len(self.input_shape) == 3:
            channels = self.input_shape[0]
        else:
            channels = self.input_shape[1]
        layers = nn.Sequential()
        layers.add_module('cnn_layer1', nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1))
        layers.add_module('cnn_activation1', nn.ReLU())
        layers.add_module('cnn_pool', nn.MaxPool2d(kernel_size=2, stride=2))
        layers.add_module('cnn_layer2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
        layers.add_module('cnn_activation2', nn.ReLU())
        layers.add_module('cnn_adaptive_pool', nn.AdaptiveMaxPool2d((32, 32)))
        layers.add_module('flatten', nn.Flatten())
        layers.add_module('cnn_fc', nn.Linear(32 * 32 * 32, self.embedder_features, bias=False))
        layers.add_module('cnn_fc_activation', nn.ReLU())
        return layers

    def _build_fc(self):
        layers = nn.Sequential()
        layers.add_module('input_layer', nn.Linear(self.linear_features, self.hidden_neurons[0], bias=False))
        layers.add_module('hidden_activation_e0', get_activation_by_name(self.hidden_activation))
        for i in range(1, len(self.hidden_neurons) - 1):
            layers.add_module(f'hidden_layer_e{i}', nn.Linear(self.hidden_neurons[i - 1], self.hidden_neurons[i], bias=False))
            layers.add_module(f'hidden_activation_e{i}', get_activation_by_name(self.hidden_activation))
            layers.add_module(f'hidden_dropout_e{i}', nn.Dropout(self.dropout_rate))
        layers.add_module('net_output', nn.Linear(self.hidden_neurons[-2], self.hidden_neurons[-1], bias=False))
        layers.add_module(f'hidden_activation_e{len(self.hidden_neurons)}', get_activation_by_name(self.hidden_activation))

        if self.use_ae:
            # Add reverse layers for the autoencoder if needed
            for j in range(len(self.hidden_neurons) - 1, 0, -1):
                layers.add_module(f'hidden_layer_d{j}', nn.Linear(self.hidden_neurons[j], self.hidden_neurons[j - 1], bias=False))
                layers.add_module(f'hidden_activation_d{j}', get_activation_by_name(self.hidden_activation))
                layers.add_module(f'hidden_dropout_d{j}', nn.Dropout(self.dropout_rate))
            layers.add_module('output_layer', nn.Linear(self.hidden_neurons[0], self.n_features, bias=False))
            layers.add_module('output_activation', get_activation_by_name(self.output_activation))

        return layers

    def forward(self, x):
        if self.feature_type == "obs":
            x = self.embedder(x)
        elif self.feature_type == "hidden_obs":
            features = self.embedder(x[0])
            x = torch.cat([features, x[1]], dim=-1)
        x = self.fc_part(x)
        return x

class DeepSVDD(BaseDetector):
    """Deep One-Class Classifier with AutoEncoder (AE) is a type of neural
        networks for learning useful data representations in an unsupervised way.
        DeepSVDD trains a neural network while minimizing the volume of a
        hypersphere that encloses the network representations of the data,
        forcing the network to extract the common factors of variation.
        Similar to PCA, DeepSVDD could be used to detect outlying objects in the
        data by calculating the distance from center
        See :cite:`ruff2018deepsvdd` for details.

        Parameters
        ----------
        n_features: int,
            Number of features in the input data.

        c: float, optional (default='forwad_nn_pass')
            Deep SVDD center, the default will be calculated based on network
            initialization first forward pass. To get repeated results set
            random_state if c is set to None.

        use_ae: bool, optional (default=False)
            The AutoEncoder type of DeepSVDD it reverse neurons from hidden_neurons
            if set to True.

        hidden_neurons : list, optional (default=[64, 32])
            The number of neurons per hidden layers. if use_ae is True, neurons
            will be reversed eg. [64, 32] -> [64, 32, 32, 64, n_features]

        hidden_activation : str, optional (default='relu')
            Activation function to use for hidden layers.
            All hidden layers are forced to use the same type of activation.
            See https://keras.io/activations/

        output_activation : str, optional (default='sigmoid')
            Activation function to use for output layer.
            See https://keras.io/activations/

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

        random_state : random_state: int, RandomState instance or None, optional
            (default=None)
            If int, random_state is the seed used by the random
            number generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.

        contamination : float in (0., 0.5), optional (default=0.1)
            The amount of contamination of the data set, i.e.
            the proportion of outliers in the data set. When fitting this is used
            to define the threshold on the decision function.

        Attributes
        ----------
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

    def __init__(self, n_features, c=None, use_ae=False, hidden_neurons=None,
                 hidden_activation='relu',
                 output_activation='sigmoid', optimizer='adam', epochs=100,
                 batch_size=32,
                 dropout_rate=0.2, l2_regularizer=0.1, feature_type="obs", validation_size=0.1,
                 preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1, input_shape=None):
        super(DeepSVDD, self).__init__(contamination=contamination)

        self.n_features = n_features
        self.c = c
        self.use_ae = use_ae
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.feature_type = feature_type
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None
        self.best_model_dict = None
        self.input_shape = input_shape

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate', include_left=True)

        # Initialize the DeepSVDD model with updated input shape
        self.model_ = InnerDeepSVDD(
            n_features=self.n_features,  # Now determined by CNN output
            use_ae=self.use_ae,
            hidden_neurons=self.hidden_neurons,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            dropout_rate=self.dropout_rate,
            l2_regularizer=self.l2_regularizer,
            feature_type=self.feature_type,
            input_shape=self.input_shape,
        )

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : list or numpy array of shape (n_samples, channels, height, width)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X_norm = self.normalization(X)

        if self.c is None:
            self.c = 0.0
            self.model_._init_c(X_norm)

        # Prepare DataLoader for batch processing
        if self.feature_type == "hidden_obs":
            dataset = TensorDataset(*X_norm, *X_norm)
        else:
            dataset = TensorDataset(X_norm, X_norm)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        best_model_dict = None
        optimizer = optimizer_dict[self.optimizer](self.model_.parameters(), weight_decay=self.l2_regularizer)
        # w_d = 1e-6 * sum([torch.linalg.norm(w) for w in self.model_.parameters()])

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0
            for batch in dataloader:
                if self.feature_type == "hidden_obs":
                    batch_x = batch[0], batch[1]
                else:
                    batch_x = batch[0]
                outputs = self.model_(batch_x)
                dist = torch.sum((outputs - self.c) ** 2, dim=-1)

                w_d = 1e-6 * sum([torch.linalg.norm(w) for w in self.model_.parameters()])

                if self.use_ae:
                    loss = torch.mean(dist) + w_d + torch.mean(torch.square(outputs - batch_x))
                else:
                    loss = torch.mean(dist) + w_d

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_dict = self.model_.state_dict()
        self.best_model_dict = best_model_dict

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on the DeepSVDD model.
        Outliers are assigned with larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, channels, height, width)
            The input samples.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        # Normalize data if pixel values are in [0, 255] range
        X = self.normalization(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X)
            dist = torch.sum((outputs - self.c) ** 2, dim=-1)
        anomaly_scores = dist.cpu().numpy()
        return anomaly_scores

    def normalization(self, X):
        if self.feature_type in ["obs", "hidden_obs"]:
            X_img = X if self.feature_type == "obs" else X[0]
            # Normalize the image data if pixel values are in the range [0, 255]
            if X_img.max() > 1:
                X_img = X_img / 255.0
            X_norm = X_img if self.feature_type == "obs" else [X_img, X[1]]
        elif self.feature_type in ["hidden", "dist"]:
            X_norm = X
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        return X_norm