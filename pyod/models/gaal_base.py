# -*- coding: utf-8 -*-
"""Base file for Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""

import math

try:
    import torch
except ImportError:
    print('please install torch first')

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        def __init__(self, latent_size, data_size):
            super(Discriminator, self).__init__()
            self.layer1 = nn.Linear(latent_size,
                                    math.ceil(math.sqrt(data_size)))
            self.layer2 = nn.Linear(math.ceil(math.sqrt(data_size)), 1)
            nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in',
                                    nonlinearity='relu')
            nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in',
                                    nonlinearity='sigmoid')

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            return x

    return Discriminator(latent_size, data_size)


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
        def __init__(self, latent_size):
            super(Generator, self).__init__()
            self.layer1 = nn.Linear(latent_size, latent_size)
            self.layer2 = nn.Linear(latent_size, latent_size)
            nn.init.eye_(self.layer1.weight)
            nn.init.eye_(self.layer2.weight)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return x

    return Generator(latent_size)
