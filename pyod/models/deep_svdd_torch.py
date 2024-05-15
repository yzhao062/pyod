import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import numpy as np

from .base import BaseDetector
from ..utils.utility import check_parameter
from ..utils.torch_utility import get_activation_by_name

class InnerDeepSVDD(nn.Module):
        def __init__(self, n_features, use_ae=False,
                  hidden_neurons=None, hidden_activation='relu',
                 output_activation='sigmoid',
                 dropout_rate=0.2, l2_regularizer=0.1):
            super(InnerDeepSVDD, self).__init__()
            self.n_features = n_features
            # self.c = c
            self.use_ae = use_ae
            self.hidden_neurons = hidden_neurons or [64, 32]
            self.hidden_activation = hidden_activation
            self.output_activation = output_activation
            self.dropout_rate = dropout_rate
            self.l2_regularizer = l2_regularizer
            self.model = self._build_model()
        
        def _init_c(self, X_norm, eps=0.1):
            intermediate_output = {}
            hook_handle = self.model._modules.get('net_output').register_forward_hook(
                lambda module, input, output: intermediate_output.update({'net_output': output})
            )
            output =self.model(X_norm)

            out = intermediate_output['net_output']
            hook_handle.remove()

            self.c = torch.mean(out, dim=0)
            self.c[(torch.abs(self.c) < eps) & (self.c < 0)] = -eps
            self.c[(torch.abs(self.c) < eps) & (self.c > 0)] = eps

        def _build_model(self):
            layers = nn.Sequential()
            layers.add_module('input_layer', nn.Linear(self.n_features, self.hidden_neurons[0], bias=False))
            for i in range(1,len(self.hidden_neurons)-1):
                layers.add_module(f'hidden_layer_e{i}', nn.Linear(self.hidden_neurons[i-1], self.hidden_neurons[i], bias=False))
                layers.add_module(f'hidden_activation_e{i}', get_activation_by_name(self.hidden_activation))
                layers.add_module(f'hidden_dropout_e{i}', nn.Dropout(self.dropout_rate))
            layers.add_module(f'net_output', nn.Linear(self.hidden_neurons[-2], self.hidden_neurons[-1], bias=False))
            
            if self.use_ae:
                for j in range(len(self.hidden_neurons)-1,1):
                    layers.add_module(f'hidden_layer_d{j}', nn.Linear(self.hidden_neurons[j], self.hidden_neurons[j-1], bias=False))
                    layers.add_module(f'hidden_dropout_d{j}', nn.Dropout(self.dropout_rate))
                layers.add_module(f'output_layer', nn.Linear(self.hidden_neurons[0], self.n_features, bias=False))
        
            return layers

        def forward(self, x):
            return self.model(x)


class DeepSVDD(BaseDetector):
    def __init__(self, n_features, c=None, use_ae=False, hidden_neurons=None, hidden_activation='relu',
                 output_activation='sigmoid', optimizer='adam', epochs=100, batch_size=32,
                 dropout_rate=0.2, l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1):
        super(DeepSVDD, self).__init__(contamination=contamination)
        
        self.n_features = n_features
        self.c = c
        self.use_ae = use_ae
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
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
        self.optimizer = optimizer_dict[optimizer]
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.model = None
        self.best_model_dict = None

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def fit(self, X, y=None):
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
        if np.min(self.hidden_neurons) > self.n_features_ and self.use_ae:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        
        # Build DeepSVDD model & fit with X
        self.model = InnerDeepSVDD(self.n_features, use_ae=self.use_ae, hidden_neurons=self.hidden_neurons,
                                   hidden_activation=self.hidden_activation, output_activation=self.output_activation,
                                   dropout_rate=self.dropout_rate, l2_regularizer=self.l2_regularizer)
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
        if self.c is None:
            self.c = 0.0
            self.model._init_c(X_norm)

        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)
        
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
        dataset = TensorDataset(X_norm, X_norm)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        best_model_dict = None

        optimizer = self.optimizer(self.model.parameters(), weight_decay=self.l2_regularizer)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                dist = torch.sum((outputs - self.c) ** 2, dim=-1)
                loss = torch.mean(dist)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_dict = self.model.state_dict()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss}")
        self.best_model_dict = best_model_dict

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
        # check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_norm)
            dist = torch.sum((outputs - self.c) ** 2, dim=-1)
        return dist.numpy()
