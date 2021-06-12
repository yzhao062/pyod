import torch

import torch.nn as nn


def get_activation_by_name(name):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }

    if name in activations.keys():
        return activations[name]

    else:
        raise ValueError(name, "is not a valid activation function")
