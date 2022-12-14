# -*- coding: utf-8 -*-
"""Using Adversarially Learned Anomaly Detection
"""
# Author: Michiel Bongaerts (but not author of the ALAD method)

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from .base_dl import _get_tensorflow_version
from ..utils.utility import check_parameter

# if tensorflow 2, import from tf directly
if _get_tensorflow_version() < 200:
    raise NotImplementedError('Model not implemented for Tensorflow version 1')
elif 200 <= _get_tensorflow_version() <= 209:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
else:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.optimizers.legacy import Adam


class ALAD(BaseDetector):
    """Adversarially Learned Anomaly Detection (ALAD). 
    Paper: https://arxiv.org/pdf/1812.02288.pdf

    See :cite:`zenati2018adversarially` for details.
    
    Parameters
    ----------
    output_activation : str, optional (default=None)
        Activation function to use for output layers for encoder and dector.
        See https://keras.io/activations/

    activation_hidden_disc : str, optional (default='tanh')
        Activation function to use for hidden layers in discrimators.
        See https://keras.io/activations/

    activation_hidden_gen : str, optional (default='tanh')
        Activation function to use for hidden layers in encoder and decoder
        (i.e. generator).
        See https://keras.io/activations/

    epochs : int, optional (default=500)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    dec_layers : list, optional (default=[5,10,25])
        List that indicates the number of nodes per hidden layer for the d
        ecoder network.
        Thus, [10,10] indicates 2 hidden layers having each 10 nodes.

    enc_layers : list, optional (default=[25,10,5])
        List that indicates the number of nodes per hidden layer for the
        encoder network.
        Thus, [10,10] indicates 2 hidden layers having each 10 nodes.

    disc_xx_layers : list, optional (default=[25,10,5])
        List that indicates the number of nodes per hidden layer for
        discriminator_xx.
        Thus, [10,10] indicates 2 hidden layers having each 10 nodes.

    disc_zz_layers : list, optional (default=[25,10,5])
        List that indicates the number of nodes per hidden layer for
        discriminator_zz.
        Thus, [10,10] indicates 2 hidden layers having each 10 nodes.

    disc_xz_layers : list, optional (default=[25,10,5])
        List that indicates the number of nodes per hidden layer for
        discriminator_xz.
        Thus, [10,10] indicates 2 hidden layers having each 10 nodes.

    learning_rate_gen: float in (0., 1), optional (default=0.001)
        learning rate of training the encoder and decoder

    learning_rate_disc: float in (0., 1), optional (default=0.001)
        learning rate of training the discriminators

    add_recon_loss: bool optional (default=False)
        add an extra loss for encoder and decoder based on the reconstruction
        error

    lambda_recon_loss: float in (0., 1), optional (default=0.1)
        if ``add_recon_loss= True``, the reconstruction loss gets multiplied
        by ``lambda_recon_loss`` and added to the total loss for the generator
         (i.e. encoder and decoder).

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

    def __init__(self, activation_hidden_gen='tanh',
                 activation_hidden_disc='tanh',
                 output_activation=None,
                 dropout_rate=0.2,
                 latent_dim=2,
                 dec_layers=[5, 10, 25],
                 enc_layers=[25, 10, 5],
                 disc_xx_layers=[25, 10, 5],
                 disc_zz_layers=[25, 10, 5],
                 disc_xz_layers=[25, 10, 5],
                 learning_rate_gen=0.0001, learning_rate_disc=0.0001,
                 add_recon_loss=False, lambda_recon_loss=0.1,
                 epochs=200,
                 verbose=0,
                 preprocessing=False,
                 add_disc_zz_loss=True, spectral_normalization=False,
                 batch_size=32, contamination=0.1):
        super(ALAD, self).__init__(contamination=contamination)

        self.activation_hidden_disc = activation_hidden_disc
        self.activation_hidden_gen = activation_hidden_gen
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.dec_layers = dec_layers
        self.enc_layers = enc_layers

        self.disc_xx_layers = disc_xx_layers
        self.disc_zz_layers = disc_zz_layers
        self.disc_xz_layers = disc_xz_layers

        self.add_recon_loss = add_recon_loss
        self.lambda_recon_loss = lambda_recon_loss
        self.add_disc_zz_loss = add_disc_zz_loss

        self.output_activation = output_activation
        self.contamination = contamination
        self.epochs = epochs
        self.learning_rate_gen = learning_rate_gen
        self.learning_rate_disc = learning_rate_disc
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.verbose = verbose
        self.spectral_normalization = spectral_normalization

        if (self.spectral_normalization == True):
            try:
                global tfa
                import tensorflow_addons as tfa
            except ModuleNotFoundError:
                # Error handling
                print(
                    'tensorflow_addons not found, cannot use spectral normalization. Install tensorflow_addons first.')
                self.spectral_normalization = False

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def _build_model(self):

        #### Decoder #####
        dec_in = Input(shape=(self.latent_dim,), name='I1')
        dec_1 = Dropout(self.dropout_rate)(dec_in)
        last_layer = dec_1

        # Store all hidden layers in dict
        dec_hl_dict = {}
        for i, l_dim in enumerate(self.dec_layers):
            layer_name = 'hl_{}'.format(i)
            dec_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                Dense(l_dim, activation=self.activation_hidden_gen)(
                    last_layer))
            last_layer = dec_hl_dict[layer_name]

        dec_out = Dense(self.n_features_, activation=self.output_activation)(
            last_layer)

        self.dec = Model(inputs=(dec_in), outputs=[dec_out])
        self.hist_loss_dec = []

        #### Encoder #####
        enc_in = Input(shape=(self.n_features_,), name='I1')
        enc_1 = Dropout(self.dropout_rate)(enc_in)
        last_layer = enc_1

        # Store all hidden layers in dict
        enc_hl_dict = {}
        for i, l_dim in enumerate(self.enc_layers):
            layer_name = 'hl_{}'.format(i)
            enc_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                Dense(l_dim, activation=self.activation_hidden_gen)(
                    last_layer))
            last_layer = enc_hl_dict[layer_name]

        enc_out = Dense(self.latent_dim, activation=self.output_activation)(
            last_layer)

        self.enc = Model(inputs=(enc_in), outputs=[enc_out])
        self.hist_loss_enc = []

        #### Discriminator_xz #####
        disc_xz_in_x = Input(shape=(self.n_features_,), name='I1')
        disc_xz_in_z = Input(shape=(self.latent_dim,), name='I2')
        disc_xz_in = tf.concat([disc_xz_in_x, disc_xz_in_z], axis=1)

        disc_xz_1 = Dropout(self.dropout_rate)(disc_xz_in)
        last_layer = disc_xz_1

        # Store all hidden layers in dict
        disc_xz_hl_dict = {}
        for i, l_dim in enumerate(self.disc_xz_layers):
            layer_name = 'hl_{}'.format(i)

            if (self.spectral_normalization == True):
                disc_xz_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                    tfa.layers.SpectralNormalization(
                        Dense(l_dim, activation=self.activation_hidden_disc))(
                        last_layer))
            else:
                disc_xz_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                    Dense(l_dim, activation=self.activation_hidden_disc)(
                        last_layer))

            last_layer = disc_xz_hl_dict[layer_name]

        disc_xz_out = Dense(1, activation='sigmoid')(last_layer)
        self.disc_xz = Model(inputs=(disc_xz_in_x, disc_xz_in_z),
                             outputs=[disc_xz_out])
        # self.hist_loss_disc_xz = []

        #### Discriminator_xx #####
        disc_xx_in_x = Input(shape=(self.n_features_,), name='I1')
        disc_xx_in_x_hat = Input(shape=(self.n_features_,), name='I2')
        disc_xx_in = tf.concat([disc_xx_in_x, disc_xx_in_x_hat], axis=1)

        disc_xx_1 = Dropout(self.dropout_rate,
                            input_shape=(self.n_features_,))(disc_xx_in)
        last_layer = disc_xx_1

        # Store all hidden layers in dict
        disc_xx_hl_dict = {}
        for i, l_dim in enumerate(self.disc_xx_layers):
            layer_name = 'hl_{}'.format(i)

            if (self.spectral_normalization == True):
                disc_xx_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                    tfa.layers.SpectralNormalization(
                        Dense(l_dim, activation=self.activation_hidden_disc))(
                        last_layer))
            else:
                disc_xx_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                    Dense(l_dim, activation=self.activation_hidden_disc)(
                        last_layer))

            last_layer = disc_xx_hl_dict[layer_name]

        disc_xx_out = Dense(1, activation='sigmoid')(last_layer)
        self.disc_xx = Model(inputs=(disc_xx_in_x, disc_xx_in_x_hat),
                             outputs=[disc_xx_out, last_layer])
        # self.hist_loss_disc_xx = []

        #### Discriminator_zz #####
        disc_zz_in_z = Input(shape=(self.latent_dim,), name='I1')
        disc_zz_in_z_prime = Input(shape=(self.latent_dim,), name='I2')
        disc_zz_in = tf.concat([disc_zz_in_z, disc_zz_in_z_prime], axis=1)

        disc_zz_1 = Dropout(self.dropout_rate,
                            input_shape=(self.n_features_,))(disc_zz_in)
        last_layer = disc_zz_1

        # Store all hidden layers in dict
        disc_zz_hl_dict = {}
        for i, l_dim in enumerate(self.disc_zz_layers):
            layer_name = 'hl_{}'.format(i)

            if (self.spectral_normalization == True):
                disc_zz_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                    tfa.layers.SpectralNormalization(
                        Dense(l_dim, activation=self.activation_hidden_disc))(
                        last_layer))
            else:
                disc_zz_hl_dict[layer_name] = Dropout(self.dropout_rate)(
                    Dense(l_dim, activation=self.activation_hidden_disc)(
                        last_layer))

            last_layer = disc_zz_hl_dict[layer_name]

        disc_zz_out = Dense(1, activation='sigmoid')(last_layer)
        self.disc_zz = Model(inputs=(disc_zz_in_z, disc_zz_in_z_prime),
                             outputs=[disc_zz_out])
        # self.hist_loss_disc_zz = []

        # Set optimizer
        opt_gen = Adam(learning_rate=self.learning_rate_gen)
        opt_disc = Adam(learning_rate=self.learning_rate_disc)

        self.dec.compile(optimizer=opt_gen)
        self.enc.compile(optimizer=opt_gen)
        self.disc_xz.compile(optimizer=opt_disc)
        self.disc_xx.compile(optimizer=opt_disc)
        self.disc_zz.compile(optimizer=opt_disc)

        self.hist_loss_disc = []
        self.hist_loss_gen = []

    def train_step(self, data):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        x_real, z_real = data

        def get_losses():
            y_true = tf.ones_like(x_real[:, [0]])
            y_fake = tf.zeros_like(x_real[:, [0]])

            # Generator
            x_gen = self.dec({'I1': z_real}, training=True)

            # Encoder
            z_gen = self.enc({'I1': x_real}, training=True)

            # Discriminatorxz
            out_truexz = self.disc_xz({'I1': x_real, 'I2': z_gen},
                                      training=True)
            out_fakexz = self.disc_xz({'I1': x_gen, 'I2': z_real},
                                      training=True)

            # Discriminatorzz
            if (self.add_disc_zz_loss == True):
                out_truezz = self.disc_zz({'I1': z_real, 'I2': z_real},
                                          training=True)
                out_fakezz = self.disc_zz({'I1': z_real, 'I2': self.enc(
                    {'I1': self.dec({'I1': z_real}, training=True)})},
                                          training=True)

                # Discriminatorxx
            out_truexx, _ = self.disc_xx({'I1': x_real, 'I2': x_real},
                                         training=True)  # self.Dxx(x_real, x_real)
            out_fakexx, _ = self.disc_xx({'I1': x_real, 'I2': self.dec(
                {'I1': self.enc({'I1': x_real}, training=True)})},
                                         training=True)

            # Losses for discriminators
            loss_dxz = cross_entropy(y_true, out_truexz) + cross_entropy(
                y_fake, out_fakexz)
            loss_dxx = cross_entropy(y_true, out_truexx) + cross_entropy(
                y_fake, out_fakexx)
            if (self.add_disc_zz_loss == True):
                loss_dzz = cross_entropy(y_true, out_truezz) + cross_entropy(
                    y_fake, out_fakezz)
                loss_disc = loss_dxz + loss_dzz + loss_dxx
            else:
                loss_disc = loss_dxz + loss_dxx

            # Losses for generator
            loss_gexz = cross_entropy(y_true, out_fakexz) + cross_entropy(
                y_fake, out_truexz)
            loss_gexx = cross_entropy(y_true, out_fakexx) + cross_entropy(
                y_fake, out_truexx)
            if (self.add_disc_zz_loss == True):
                loss_gezz = cross_entropy(y_true, out_fakezz) + cross_entropy(
                    y_fake, out_truezz)
                cycle_consistency = loss_gezz + loss_gexx
                loss_gen = loss_gexz + cycle_consistency
            else:
                cycle_consistency = loss_gexx
                loss_gen = loss_gexz + cycle_consistency

            if (self.add_recon_loss == True):
                # Extra recon loss
                x_recon = self.dec(
                    {'I1': self.enc({'I1': x_real}, training=True)})
                loss_recon = tf.reduce_mean((x_real - x_recon) ** 2)
                loss_gen += loss_recon * self.lambda_recon_loss

            return loss_disc, loss_gen

        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_xx_tape, tf.GradientTape() as disc_xz_tape, tf.GradientTape() as disc_zz_tape:
            loss_disc, loss_gen = get_losses()

        self.hist_loss_disc.append(np.float64(loss_disc.numpy()))
        self.hist_loss_gen.append(np.float64(loss_gen.numpy()))

        gradients_dec = dec_tape.gradient(loss_gen,
                                          self.dec.trainable_variables)
        self.dec.optimizer.apply_gradients(
            zip(gradients_dec, self.dec.trainable_variables))

        gradients_enc = enc_tape.gradient(loss_gen,
                                          self.enc.trainable_variables)
        self.enc.optimizer.apply_gradients(
            zip(gradients_enc, self.enc.trainable_variables))

        gradients_disc_xx = disc_xx_tape.gradient(loss_disc,
                                                  self.disc_xx.trainable_variables)
        self.disc_xx.optimizer.apply_gradients(
            zip(gradients_disc_xx, self.disc_xx.trainable_variables))

        if (self.add_disc_zz_loss == True):
            gradients_disc_zz = disc_zz_tape.gradient(loss_disc,
                                                      self.disc_zz.trainable_variables)
            self.disc_zz.optimizer.apply_gradients(
                zip(gradients_disc_zz, self.disc_zz.trainable_variables))

        gradients_disc_xz = disc_xz_tape.gradient(loss_disc,
                                                  self.disc_xz.trainable_variables)
        self.disc_xz.optimizer.apply_gradients(
            zip(gradients_disc_xz, self.disc_xz.trainable_variables))

    def plot_learning_curves(self, start_ind=0, window_smoothening=10):
        fig = plt.figure(figsize=(12, 5))

        l_gen = pd.Series(self.hist_loss_gen[start_ind:]).rolling(
            window_smoothening).mean()
        l_disc = pd.Series(self.hist_loss_disc[start_ind:]).rolling(
            window_smoothening).mean()

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(range(len(l_gen)), l_gen, )
        ax.set_title('Generator')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Iter')

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(range(len(l_disc)), l_disc)
        ax.set_title('Discriminator(s)')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Iter')

        plt.show()

    def fit(self, X, y=None, noise_std=0.1):
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

        # Get number of sampels and features from train set
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]
        self._build_model()

        # Apply data scaling or not
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        for n in range(self.epochs):
            if ((n % 50 == 0) and (n != 0) and (self.verbose == 1)):
                print('Train iter:{}'.format(n))

            # Shuffle train 
            np.random.shuffle(X_norm)

            X_train_sel = X_norm[0: min(self.batch_size, self.n_samples_), :]
            latent_noise = np.random.normal(0, 1, (
                X_train_sel.shape[0], self.latent_dim))
            X_train_sel += np.random.normal(0, noise_std,
                                            size=X_train_sel.shape)
            self.train_step(
                (np.float32(X_train_sel), np.float32(latent_noise)))

            # Predict on X itself and calculate the the outlier scores.
        # Note, X_norm was shuffled and needs to be recreated
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.get_outlier_scores(X_norm)
        self.decision_scores_ = pred_scores
        self._process_decision_scores()
        return self

    def train_more(self, X, epochs=100, noise_std=0.1):
        """This function allows the researcher to perform extra training instead of the fixed number determined
        by the fit() function.
        """

        # fit() should have been called first
        check_is_fitted(self, ['decision_scores_'])

        # Apply data scaling or not
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        for n in range(epochs):
            if ((n % 50 == 0) and (n != 0) and (self.verbose == 1)):
                print('Train iter:{}'.format(n))

            # Shuffle train 
            np.random.shuffle(X_norm)

            X_train_sel = X_norm[0: min(self.batch_size, self.n_samples_), :]
            latent_noise = np.random.normal(0, 1, (
                X_train_sel.shape[0], self.latent_dim))
            X_train_sel += np.random.normal(0, noise_std,
                                            size=X_train_sel.shape)
            self.train_step(
                (np.float32(X_train_sel), np.float32(latent_noise)))

            # Predict on X itself and calculate the the outlier scores.
        # Note, X_norm was shuffled and needs to be recreated
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.get_outlier_scores(X_norm)
        self.decision_scores_ = pred_scores
        self._process_decision_scores()
        return self

    def get_outlier_scores(self, X_norm):

        X_enc = self.enc({'I1': X_norm}).numpy()
        X_enc_gen = self.dec({'I1': X_enc}).numpy()

        _, act_layer_xx = self.disc_xx({'I1': X_norm, 'I2': X_norm},
                                       training=False)
        act_layer_xx = act_layer_xx.numpy()
        _, act_layer_xx_enc_gen = self.disc_xx({'I1': X_norm, 'I2': X_enc_gen},
                                               training=False)
        act_layer_xx_enc_gen = act_layer_xx_enc_gen.numpy()
        outlier_scores = np.mean(
            np.abs((act_layer_xx - act_layer_xx_enc_gen) ** 2), axis=1)

        return outlier_scores

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

        # Predict on X 
        pred_scores = self.get_outlier_scores(X_norm)
        return pred_scores
