# -*- coding: utf-8 -*-
"""Base file for Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""

from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: create a base class for so_gaal and mo_gaal
def create_discriminator(latent_size, data_size):  # pragma: no cover
    """Create the discriminator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator.

    data_size : int
        Size of the input data.

    Returns
    -------
    D : PyTorch model
        Returns a model.
    """

    class Discriminator(nn.Module):
        def __init__(self, latent_size, data_size):
            super(Discriminator, self).__init__()
            self.layer1 = nn.Linear(latent_size, int(math.ceil(math.sqrt(data_size))), bias=True)
            self.output = nn.Linear(int(math.ceil(math.sqrt(data_size))), 1, bias=True)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = torch.sigmoid(self.output(x))
            return x

    return Discriminator(latent_size, data_size)


def create_generator(latent_size):  # pragma: no cover
    """Create the generator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator

    Returns
    -------
    D : PyTorch model
        Returns a model.
    """

    class Generator(nn.Module):
        def __init__(self, latent_size):
            super(Generator, self).__init__()
            self.layer1 = nn.Linear(latent_size, latent_size, bias=True)
            self.layer2 = nn.Linear(latent_size, latent_size, bias=True)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return x

    return Generator(latent_size)
