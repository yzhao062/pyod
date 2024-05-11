import torch
import torch.nn as nn
import math

def create_discriminator(latent_size, data_size):
    """Create the discriminator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator.

    data_size : int
        Size of the input data.

    Returns
    -------
    D : PyTorch model object
        Returns a model object.
    """

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.fc1 = nn.Linear(latent_size, int(math.ceil(math.sqrt(data_size))))
            self.fc2 = nn.Linear(int(math.ceil(math.sqrt(data_size))), 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x

    return Discriminator()

def create_generator(latent_size):
    """Create the generator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator

    Returns
    -------
    D : PyTorch model object
        Returns a model object.
    """

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.fc1 = nn.Linear(latent_size, latent_size)
            self.fc2 = nn.Linear(latent_size, latent_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return x

    return Generator()
