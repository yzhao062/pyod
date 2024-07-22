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

# Author: Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause


try:
    import torch
except ImportError:
    print('please install torch first')

import torch
from torch import nn

from .base_dl import BaseDeepLearningDetector
from ..utils.stat_models import pairwise_distances_no_broadcast
from ..utils.torch_utility import LinearBlock, get_criterion_by_name


def vae_loss(x, x_recon, z_mu, z_logvar, beta=1.0, capacity=0.0):
    """Compute the loss of VAE

    Parameters
    ----------
    x : torch.Tensor, shape (n_samples, n_features)
        The input data.

    x_recon : torch.Tensor, shape (n_samples, n_features)
        The reconstructed data.

    z_mu : torch.Tensor, shape (n_samples, latent_dim)
        The mean of the latent distribution.

    z_logvar : torch.Tensor, shape (n_samples, latent_dim)
        The log variance of the latent distribution.

    beta : float, optional (default=1.0)
        The weight of KL divergence.

    capacity : float, optional (default=0.0)
        The maximum capacity of a loss bottleneck.

    Returns
    -------
    loss : torch.Tensor, shape (n_samples,)
        The loss of VAE.
    """
    # Reconstruction loss
    recon_loss = get_criterion_by_name('mse')(x_recon, x)

    # KL divergence
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - torch.exp(z_logvar),
                         dim=1), dim=0)
    kl_loss = torch.clamp(kl_loss, min=0, max=capacity)

    return recon_loss + beta * kl_loss


class VAE(BaseDeepLearningDetector):
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
    VAE_loss = Reconstruction_loss + beta * KL_loss

    Reference
    See :cite:`burgess2018understanding` Burges et al
    'Understanding disentangling in beta-VAE'
    https://arxiv.org/pdf/1804.03599.pdf for details.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, 
        i.e. the proportion of outliers in the data set. 
        Used when fitting to define the threshold on the decision function.

    preprocessing : bool, optional (default=True)
        If True, apply the preprocessing procedure before training models.

    lr : float, optional (default=1e-3)
        The initial learning rate for the optimizer.

    epoch_num : int, optional (default=30)
        The number of epochs for training.

    batch_size : int, optional (default=32)
        The batch size for training.

    optimizer_name : str, optional (default='adam')
        The name of theoptimizer used to train the model.

    device : str, optional (default=None)
        The device to use for the model. If None, it will be decided
        automatically. If you want to use MPS, set it to 'mps'.

    random_state : int, optional (default=42)
        The random seed for reproducibility.

    use_compile : bool, optional (default=False)
        Whether to compile the model.
        If True, the model will be compiled before training.
        This is only available for
        PyTorch version >= 2.0.0. and Python < 3.12.

    compile_mode : str, optional (default='default')
        The mode to compile the model.
        Can be either “default”, “reduce-overhead”,
        “max-autotune” or “max-autotune-no-cudagraphs”.
        See https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile for details.

    verbose : int, optional (default=1)
        Verbosity mode.
        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

    optimizer_params : dict, optional (default={'weight_decay': 1e-5})
        Additional parameters for the optimizer.
        For example, `optimizer_params={'weight_decay': 1e-5}`.

    beta : float, optional (default=1.0)
        Coefficient of beta VAE. The weight of KL divergence.
        Default is regular VAE.

    capacity : float, optional (default=0.0)
        The maximum capacity of a loss bottleneck.

    encoder_neuron_list : list, optional (default=[128, 64, 32])
        The number of neurons per hidden layers in encoder.
        So the encoder has the structure as [feature_size, 128, 64, 32, latent_dim].

    decoder_neuron_list : list, optional (default=[32, 64, 128])
        The number of neurons per hidden layers in decoder.
        So the decoder has the structure as [latent_dim, 32, 64, 128, feature_size].

    latent_dim : int, optional (default=2)
        The dimension of latent space.

    hidden_activation_name : str, optional (default='relu')
        The activation function used in hidden layers.

    output_activation_name : str, optional (default='sigmoid')
        The activation function used in output layer.

    batch_norm : boolean, optional (default=False)
        Whether to apply Batch Normalization,
        See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying VAE model.

    optimizer : torch.optim
        The optimizer used to train the model.

    criterion : python function
        The loss function used to train the model.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

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

    def __init__(self, contamination=0.1, preprocessing=True,
                 lr=1e-3, epoch_num=30, batch_size=32,
                 optimizer_name='adam',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 optimizer_params: dict = {'weight_decay': 1e-5},
                 beta=1.0, capacity=0.0,
                 encoder_neuron_list=[128, 64, 32],
                 decoder_neuron_list=[32, 64, 128],
                 latent_dim=2,
                 hidden_activation_name='relu',
                 output_activation_name='sigmoid',
                 batch_norm=False, dropout_rate=0.2):
        super(VAE, self).__init__(contamination=contamination,
                                  preprocessing=preprocessing,
                                  lr=lr, epoch_num=epoch_num,
                                  batch_size=batch_size,
                                  optimizer_name=optimizer_name,
                                  loss_func=vae_loss,
                                  device=device, random_state=random_state,
                                  use_compile=use_compile,
                                  compile_mode=compile_mode,
                                  verbose=verbose,
                                  optimizer_params=optimizer_params)
        self.beta = beta
        self.capacity = capacity
        self.encoder_neuron_list = encoder_neuron_list
        self.decoder_neuron_list = decoder_neuron_list
        self.latent_dim = latent_dim
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

    def build_model(self):
        self.model = VAEModel(self.feature_size,
                              encoder_neuron_list=self.encoder_neuron_list,
                              decoder_neuron_list=self.decoder_neuron_list,
                              latent_dim=self.latent_dim,
                              hidden_activation_name=self.hidden_activation_name,
                              output_activation_name=self.output_activation_name,
                              batch_norm=self.batch_norm,
                              dropout_rate=self.dropout_rate)

    def training_forward(self, batch_data):
        x = batch_data
        x = x.to(self.device)
        self.optimizer.zero_grad()
        x_recon, z_mu, z_logvar = self.model(x)
        loss = self.criterion(x, x_recon, z_mu, z_logvar,
                              beta=self.beta, capacity=self.capacity)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluating_forward(self, batch_data):
        x = batch_data
        x_gpu = x.to(self.device)
        x_recon, _, _ = self.model(x_gpu)
        score = pairwise_distances_no_broadcast(x.numpy(),
                                                x_recon.cpu().numpy())
        return score


class VAEModel(nn.Module):
    def __init__(self,
                 feature_size,
                 encoder_neuron_list=[128, 64, 32],
                 decoder_neuron_list=[32, 64, 128],
                 latent_dim=2,
                 hidden_activation_name='relu',
                 output_activation_name='sigmoid',
                 batch_norm=False, dropout_rate=0.2):
        super(VAEModel, self).__init__()

        self.feature_size = feature_size
        self.encoder_neuron_list = encoder_neuron_list
        self.decoder_neuron_list = decoder_neuron_list
        self.latent_dim = latent_dim
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.encoder_mu = nn.Linear(encoder_neuron_list[-1], latent_dim)
        self.encoder_logvar = nn.Linear(encoder_neuron_list[-1], latent_dim)

    def _build_encoder(self):
        encoder_layers = []
        last_neuron_size = self.feature_size
        for neuron_size in self.encoder_neuron_list:
            encoder_layers.append(LinearBlock(last_neuron_size, neuron_size,
                                              activation_name=self.hidden_activation_name,
                                              batch_norm=self.batch_norm,
                                              dropout_rate=self.dropout_rate))
            last_neuron_size = neuron_size
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self):
        decoder_layers = []
        last_neuron_size = self.latent_dim
        for neuron_size in self.decoder_neuron_list:
            decoder_layers.append(LinearBlock(last_neuron_size, neuron_size,
                                              activation_name=self.hidden_activation_name,
                                              batch_norm=self.batch_norm,
                                              dropout_rate=self.dropout_rate))
            last_neuron_size = neuron_size
        decoder_layers.append(LinearBlock(last_neuron_size, self.feature_size,
                                          activation_name=self.output_activation_name,
                                          batch_norm=False,
                                          dropout_rate=0))
        return nn.Sequential(*decoder_layers)

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decode(z)
        return x_recon, z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(std.device)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        z_mu = self.encoder_mu(h)
        z_logvar = self.encoder_logvar(h)
        return z_mu, z_logvar

    def decode(self, z):
        return self.decoder(z)
