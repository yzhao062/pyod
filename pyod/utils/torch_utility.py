# -*- coding: utf-8 -*-
"""Utility functions for PyTorch models
"""
# Author: Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause

import torch
import torch.nn as nn


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, mean=None, std=None, eps=1e-8,
                 X_dtype=torch.float32, y_dtype=torch.float32,
                 return_idx=False):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std
        self.eps = eps
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.return_idx = return_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / (self.std + self.eps)

        if self.y is not None:
            if self.return_idx:
                return torch.as_tensor(sample, dtype=self.X_dtype), \
                    torch.as_tensor(self.y[idx], dtype=self.y_dtype), idx
            else:
                return torch.as_tensor(sample, dtype=self.X_dtype), \
                    torch.as_tensor(self.y[idx], dtype=self.y_dtype)
        else:
            if self.return_idx:
                return torch.as_tensor(sample, dtype=self.X_dtype), idx
            else:
                return torch.as_tensor(sample, dtype=self.X_dtype)


class LinearBlock(nn.Module):
    """
    Linear block with activation and batch normalization

    Parameters
    ----------
    in_features : int
        Number of input features
        
    out_features : int
        Number of output features.

    has_act : bool, optional (default=True)
        If True, apply activation function after linear layer.

    activation_name : str, optional (default='relu')
        Activation function name. Available functions: 
        'elu', 'leaky_relu', 'relu', 'sigmoid',
        'softmax', 'softplus', 'tanh'.

    batch_norm : bool, optional (default=True)
        If True, apply batch normalization after activation function if `has_act` is True,
        or after linear layer if `has_act` is False.
        The following four parameters are used only if `batch_norm` is True.
        See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#batchnorm1d for details.

    bn_eps : float, optional (default=1e-5)
        A value added to the denominator for numerical stability

    bn_momentum : float, optional (default=0.1)
        The value used for the running_mean and running_var computation. 
        Can be set to None for cumulative moving average (i.e. simple average)

    bn_affine : bool, optional (default=True)
        A boolean value that when set to 'True', this module has learnable affine parameters.

    bn_track_running_stats : bool, optional (default=True)
        Batch normalization track_running_stats.

    dropout_rate : float, optional (default=0)
        The probability of an element to be zeroed.
        See https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#dropout for details.

    init_type : str, optional (default='kaiming_uniform')
        Initialization type.
        Available types: 'uniform', 'normal', 'constant', 'ones', 'zeros', 'eye', 'dirac',
        'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'trunc_normal',
        'orthogonal', 'sparse'.
        See https://pytorch.org/docs/stable/nn.init.html for details.

    inplace : bool, optional (default=False)
        If set to True, activation function and dropout are applied in-place.

    activation_params : dict, optional (default=None)
        Additional parameters for activation function.
        For example, `activation_params={
            'elu_alpha': 1.0, 
            'leaky_relu_negative_slope': 0.01}`.

    init_params : dict, optional (default=None)
        Additional parameters for initialization function.
        For example, `init_params={
            'uniform_a': 0.0, 
            'uniform_b': 1.0}`.
    """

    def __init__(self, in_features, out_features,
                 has_act=True, activation_name='relu',
                 batch_norm=False, bn_eps=1e-5, bn_momentum=0.1,
                 bn_affine=True, bn_track_running_stats=True,
                 dropout_rate=0,
                 init_type='kaiming_uniform',
                 inplace=False,
                 activation_params: dict = {},
                 init_params: dict = {}):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.has_act = has_act
        if has_act:
            # only use the variable about activation function in **kwargs
            self.activation = get_activation_by_name(activation_name,
                                                     inplace=inplace,
                                                     **activation_params)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features, eps=bn_eps,
                                     momentum=bn_momentum, affine=bn_affine,
                                     track_running_stats=bn_track_running_stats)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=inplace)
        init_weights(layer=self.linear, name=init_type, **init_params)

    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.has_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x


def get_activation_by_name(name, inplace=False,
                           elu_alpha=1.0,
                           leaky_relu_negative_slope=0.01,
                           softmax_dim=None,
                           softplus_beta=1.0, softplus_threshold=20.0):
    """
    Get activation function by name

    Parameters
    ----------
    name : str
        Activation function name. Available functions: 
        'elu', 'leaky_relu', 'relu', 'sigmoid',
        'softmax', 'softplus', 'tanh'.

    inplace : bool, optional (default=False)
        If set to True, do the operation in-place.

    elu_alpha : float, optional (default=1.0)
        The alpha value for the ELU formulation.
        See https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#elu for details.

    leaky_relu_negative_slope : float, optional (default=0.01)
        Controls the angle of the negative slope (which is used for negative inputs values).
        See https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#leakyrelu for details.

    softmax_dim : int, optional (default=None)
        A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        See https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#softmax for details.

    softplus_beta : float, optional (default=1.0)
        The beta value for the Softplus formulation.
        See https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#softplus for details.

    softplus_threshold : float, optional (default=20.0)
        Values above this revert to a linear function
        See https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#softplus for details.

    Returns
    -------
    activation : torch.nn.Module
        Activation function module
    """
    activation_dict = {
        'elu': nn.ELU(alpha=elu_alpha, inplace=inplace),
        'leaky_relu': nn.LeakyReLU(negative_slope=leaky_relu_negative_slope,
                                   inplace=inplace),
        'relu': nn.ReLU(inplace=inplace),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(dim=softmax_dim),
        'softplus': nn.Softplus(beta=softplus_beta,
                                threshold=softplus_threshold),
        'tanh': nn.Tanh()
    }

    if name in activation_dict.keys():
        return activation_dict[name]

    else:
        raise ValueError(f"{name} is not a valid activation.")


def get_optimizer_by_name(model, name, lr=1e-3, weight_decay=0,
                          adam_eps=1e-8,
                          sgd_momentum=0, sgd_nesterov=False):
    """
    Get optimizer by name

    Parameters
    ----------
    model : torch.nn.Module
        Model to be optimized.

    name : str
        Optimizer name. Available optimizers: 'adam', 'sgd'.
        See https://pytorch.org/docs/stable/optim.html for details.

    lr : float, optional (default=1e-3)
        Learning rate.

    weight_decay : float, optional (default=0)
        Weight decay (L2 penalty).

    adam_eps : float, optional (default=1e-8)
        Term added to the denominator to improve numerical stability.
        See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam for details.

    sgd_momentum : float, optional (default=0)
        Momentum factor in SGD.
        See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD for details.

    sgd_nesterov : bool, optional (default=False)
        Enables Nesterov momentum.
        See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD for details.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        Optimizer
    """
    optimizer_dict = {
        'adam': torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay, eps=adam_eps),
        'sgd': torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=sgd_momentum,
                               weight_decay=weight_decay,
                               nesterov=sgd_nesterov)
    }

    if name in optimizer_dict.keys():
        return optimizer_dict[name]

    else:
        raise ValueError(f"{name} is not a valid optimizer.")


def get_criterion_by_name(name, reduction='mean',
                          bce_weight=None):
    """
    Get criterion by name

    Parameters
    ----------
    name : str
        Loss function name. Available functions: 'mse', 'mae', 'bce'.
        See https://pytorch.org/docs/stable/nn.html#loss-functions for details.

    reduction : str, optional (default='mean')
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied, 
        'mean': the sum of the output will be divided by the number of elements in the output, 
        'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, 
            and in the meantime, specifying either of those two args will override reduction. Default: 'mean'
        See https://pytorch.org/docs/stable/nn.html#loss-functions for details.

    bce_weight : torch.Tensor, optional (default=None)
        A manual rescaling weight given to the loss of each batch element.
        See https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss for details.

    Returns
    -------
    criterion : torch.nn.Module
        Criterion module.
    """
    criterion_dict = {
        'mse': nn.MSELoss(reduction=reduction),
        'mae': nn.L1Loss(reduction=reduction),
        'bce': nn.BCELoss(reduction=reduction, weight=bce_weight)
    }

    if name in criterion_dict.keys():
        return criterion_dict[name]

    else:
        raise ValueError(f"{name} is not a valid criterion.")


def init_weights(layer, name='kaiming_uniform',
                 uniform_a=0.0, uniform_b=1.0,
                 normal_mean=0.0, normal_std=1.0,
                 constant_val=0.0,
                 xavier_gain=1.0,
                 kaiming_a=0, kaiming_mode='fan_in',
                 kaiming_nonlinearity='leaky_relu',
                 trunc_mean=0.0, trunc_std=1.0, trunc_a=-2, trunc_b=2,
                 orthogonal_gain=1.0,
                 sparse_sparsity=None, sparse_std=0.01, sparse_generator=None):
    """
    Initialize weights for a layer

    Parameters
    ----------
    layer : torch.nn.Module
        Layer to be initialized.

    name : str, optional (default='kaiming_uniform')
        Initialization type.
        Available types: 'uniform', 'normal', 'constant', 'ones', 'zeros', 'eye', 'dirac',
        'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'trunc_normal',
        'orthogonal', 'sparse'.
        See https://pytorch.org/docs/stable/nn.init.html for details.

    uniform_a : float, optional (default=0.0)
        The lower bound for the uniform distribution.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_ for details.

    uniform_b : float, optional (default=1.0)
        The upper bound for the uniform distribution.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_ for details.

    normal_mean : float, optional (default=0.0)
        The mean of the normal distribution.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_ for details.

    normal_std : float, optional (default=1.0)
        The standard deviation of the normal distribution.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_ for details.

    constant_val : float, optional (default=0.0)
        The value to fill the tensor with.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.constant_ for details.

    xavier_gain : float, optional (default=1.0)
        An optional scaling factor.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_ 
        and https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_ for details.

    kaiming_a : float, optional (default=0)
        The negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_ 
        and https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_ for details.

    kaiming_mode : str, optional (default='fan_in')
        The mode for kaiming initialization. Available modes: 'fan_in', 'fan_out'.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_ 
        and https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_ for details.

    kaiming_nonlinearity : str, optional (default='leaky_relu')
        The non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu'.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_ 
        and https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_ for details.

    trunc_mean : float, optional (default=0.0)
        The mean value of the truncated normal distribution.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_ for details.

    trunc_std : float, optional (default=1.0)
        The standard deviation of the truncated normal distribution.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_ for details.

    trunc_a : float, optional (default=-2)
        The minimum cutoff value.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_ for details.

    trunc_b : float, optional (default=2)
        The maximum cutoff value.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_ for details.

    orthogonal_gain : float, optional (default=1.0)
        The optional scaling factor
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_ for details.

    sparse_sparsity : float, optional (default=None)
        This parameter must be provided if used!
        The fraction of elements in each column to be set to zero.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.sparse_ for details.

    sparse_std : float, optional (default=0.01)
        The standard deviation of the normal distribution used to generate the non-zero values
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.sparse_ for details.

    sparse_generator : Optional[Generator] (default=None)
        The torch Generator to sample from.
        See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.sparse_ for details.
    """
    init_name_dict = {
        'uniform': nn.init.uniform_,
        'normal': nn.init.normal_,
        'constant': nn.init.constant_,
        'ones': nn.init.ones_,
        'zeros': nn.init.zeros_,
        'eye': nn.init.eye_,
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'trunc_normal': nn.init.trunc_normal_,
        'orthogonal': nn.init.orthogonal_,
        'sparse': nn.init.sparse_
    }

    if name in init_name_dict.keys():
        if name == 'uniform':
            init_name_dict[name](layer.weight, a=uniform_a, b=uniform_b)
        elif name == 'normal':
            init_name_dict[name](layer.weight, mean=normal_mean,
                                 std=normal_std)
        elif name == 'constant':
            init_name_dict[name](layer.weight, val=constant_val)
        elif name == 'ones':
            init_name_dict[name](layer.weight)
        elif name == 'zeros':
            init_name_dict[name](layer.weight)
        elif name == 'eye':
            init_name_dict[name](layer.weight)
        elif name == 'xavier_uniform':
            init_name_dict[name](layer.weight, gain=xavier_gain)
        elif name == 'xavier_normal':
            init_name_dict[name](layer.weight, gain=xavier_gain)
        elif name == 'kaiming_uniform':
            init_name_dict[name](layer.weight, a=kaiming_a, mode=kaiming_mode,
                                 nonlinearity=kaiming_nonlinearity)
        elif name == 'kaiming_normal':
            init_name_dict[name](layer.weight, a=kaiming_a, mode=kaiming_mode,
                                 nonlinearity=kaiming_nonlinearity)
        elif name == 'trunc_normal':
            init_name_dict[name](layer.weight, mean=trunc_mean, std=trunc_std,
                                 a=trunc_a, b=trunc_b)
        elif name == 'orthogonal':
            init_name_dict[name](layer.weight, gain=orthogonal_gain)
        elif name == 'sparse':
            init_name_dict[name](layer.weight, sparsity=sparse_sparsity,
                                 std=sparse_std)
    else:
        raise ValueError(f"{name} is not a valid initialization type.")
