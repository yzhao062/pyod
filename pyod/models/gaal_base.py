# -*- coding: utf-8 -*-
"""Base file for Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""
# Author: Winston Li <jk_zhengli@hotmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import math

from .base_dl import _get_tensorflow_version

# if tensorflow 2, import from tf directly
if _get_tensorflow_version() <= 200:
    import keras
    from keras.layers import Input, Dense
    from keras.models import Sequential, Model
else:
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Sequential, Model


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
    D : Keras model() object
        Returns a model() object.
    """

    dis = Sequential()
    dis.add(Dense(int(math.ceil(math.sqrt(data_size))),
                  input_dim=latent_size, activation='relu',
                  kernel_initializer=keras.initializers.VarianceScaling(
                      scale=1.0, mode='fan_in', distribution='normal',
                      seed=None)))
    dis.add(Dense(1, activation='sigmoid',
                  kernel_initializer=keras.initializers.VarianceScaling(
                      scale=1.0, mode='fan_in', distribution='normal',
                      seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)


def create_generator(latent_size):  # pragma: no cover
    """Create the generator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator

    Returns
    -------
    D : Keras model() object
        Returns a model() object.
    """

    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu',
                  kernel_initializer=keras.initializers.Identity(
                      gain=1.0)))
    gen.add(Dense(latent_size, activation='relu',
                  kernel_initializer=keras.initializers.Identity(
                      gain=1.0)))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)
