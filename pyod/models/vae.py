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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..utils.utility import check_parameter
from ..utils.stat_models import pairwise_distances_no_broadcast

from .base import BaseDetector
from .base_dl import _get_tensorflow_version

# if tensorflow 2, import from tf directly
if _get_tensorflow_version() == 1:
    from keras.models import Model
    from keras.layers import Lambda, Input, Dense, Dropout
    from keras.regularizers import l2
    from keras.losses import mse, binary_crossentropy
    from keras import backend as K
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Lambda, Input, Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.losses import mse, binary_crossentropy
    from tensorflow.keras import backend as K


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

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    loss : str or obj, optional (default=keras.losses.mean_squared_error
        String (name of objective function) or objective function.
        See https://keras.io/losses/

    gamma : float, optional (default=1.0)
        Coefficient of beta VAE regime.
        Default is regular VAE.

    capacity : float, optional (default=0.0)
        Maximum capacity of a loss bottle neck.

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

    verbose : int, optional (default=1)
        verbose mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

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

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
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
                 latent_dim=2, hidden_activation='relu',
                 output_activation='sigmoid', loss=mse, optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1,
                 gamma=1.0, capacity=0.0):
        super(VAE, self).__init__(contamination=contamination)
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.capacity = capacity

        # default values
        if self.encoder_neurons is None:
            self.encoder_neurons = [128, 64, 32]

        if self.decoder_neurons is None:
            self.decoder_neurons = [32, 64, 128]

        self.encoder_neurons_ = self.encoder_neurons
        self.decoder_neurons_ = self.decoder_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def sampling(self, args):
        """Reparametrisation by sampling from Gaussian, N(0,I)
        To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
        with latent variables z: z = z_mean + sqrt(var) * epsilon

        Parameters
        ----------
        args : tensor
            Mean and log of variance of Q(z|X).
    
        Returns
        -------
        z : tensor
            Sampled latent variable.
        """

        z_mean, z_log = args
        batch = K.shape(z_mean)[0]  # batch size
        dim = K.int_shape(z_mean)[1]  # latent dimension
        epsilon = K.random_normal(shape=(batch, dim))  # mean=0, std=1.0

        return z_mean + K.exp(0.5 * z_log) * epsilon

    def vae_loss(self, inputs, outputs, z_mean, z_log):
        """ Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """

        reconstruction_loss = self.loss(inputs, outputs)
        reconstruction_loss *= self.n_features_
        kl_loss = 1 + z_log - K.square(z_mean) - K.exp(z_log)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        kl_loss = self.gamma * K.abs(kl_loss - self.capacity)

        return K.mean(reconstruction_loss + kl_loss)

    def _build_model(self):
        """Build VAE = encoder + decoder + vae_loss"""

        # Build Encoder
        inputs = Input(shape=(self.n_features_,))
        # Input layer
        layer = Dense(self.n_features_, activation=self.hidden_activation)(
            inputs)
        # Hidden layers
        for neurons in self.encoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation,
                          activity_regularizer=l2(self.l2_regularizer))(layer)
            layer = Dropout(self.dropout_rate)(layer)
        # Create mu and sigma of latent variables
        z_mean = Dense(self.latent_dim)(layer)
        z_log = Dense(self.latent_dim)(layer)
        # Use parametrisation sampling
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))(
            [z_mean, z_log])
        # Instantiate encoder
        encoder = Model(inputs, [z_mean, z_log, z])
        if self.verbose >= 1:
            encoder.summary()

        # Build Decoder
        latent_inputs = Input(shape=(self.latent_dim,))
        # Latent input layer
        layer = Dense(self.latent_dim, activation=self.hidden_activation)(
            latent_inputs)
        # Hidden layers
        for neurons in self.decoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation)(layer)
            layer = Dropout(self.dropout_rate)(layer)
        # Output layer
        outputs = Dense(self.n_features_, activation=self.output_activation)(
            layer)
        # Instatiate decoder
        decoder = Model(latent_inputs, outputs)
        if self.verbose >= 1:
            decoder.summary()
        # Generate outputs
        outputs = decoder(encoder(inputs)[2])

        # Instantiate VAE
        vae = Model(inputs, outputs)
        vae.add_loss(self.vae_loss(inputs, outputs, z_mean, z_log))
        vae.compile(optimizer=self.optimizer)
        if self.verbose >= 1:
            vae.summary()
        return vae

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
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
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
        pred_scores = self.model_.predict(X_norm)
        return pairwise_distances_no_broadcast(X_norm, pred_scores)
