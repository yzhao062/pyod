# -*- coding: utf-8 -*-
"""Base file for Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""

from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import math


def create_discriminator(latent_size, data_size):
    """
    Create the discriminator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator.
    data_size : int
        Size of the input data.

    Returns
    -------
    discriminator : torch.nn.Module
        A PyTorch model of the discriminator.
    """

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(latent_size, int(math.ceil(math.sqrt(data_size))), bias=True),
                nn.ReLU(),
                nn.Linear(int(math.ceil(math.sqrt(data_size))), 1, bias=True),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    return Discriminator()


def create_generator(latent_size):
    """
    Create the generator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator.

    Returns
    -------
    generator : torch.nn.Module
        A PyTorch model of the generator.
    """

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(latent_size, latent_size, bias=True),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size, bias=True),
                nn.ReLU()
            )

        def forward(self, x):
            return self.network(x)

    return Generator()
