# -*- coding: utf-8 -*-

"""Single-Objective Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""
# Author: Sihan Chen <schen976@usc.edu>, Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause

import math

try:
    import torch
except ImportError:
    print('please install torch first')

import torch
import torch.nn as nn

from .base_dl import BaseDeepLearningDetector
from ..utils.torch_utility import LinearBlock, get_optimizer_by_name


class SO_GAAL(BaseDeepLearningDetector):
    """Single-Objective Generative Adversarial Active Learning.

    SO-GAAL directly generates informative potential outliers to assist the
    classifier in describing a boundary that can separate outliers from normal
    data effectively. Moreover, to prevent the generator from falling into the
    mode collapsing problem, the network structure of SO-GAAL is expanded from
    a single generator (SO-GAAL) to multiple generators with different
    objectives (MO-GAAL) to generate a reasonable reference distribution for
    the whole dataset.

    Parameters
    ----------
    
    """

    def __init__(self, contamination=0.1, preprocessing=True,
                 epoch_num=100,
                 criterion_name='bce',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 lr_d=0.01, lr_g=0.0001, momentum=0.9):
        super(SO_GAAL, self).__init__(contamination=contamination,
                                      preprocessing=preprocessing,
                                      epoch_num=epoch_num,
                                      criterion_name=criterion_name,
                                      device=device, random_state=random_state,
                                      use_compile=use_compile,
                                      compile_mode=compile_mode,
                                      verbose=verbose)
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.momentum = momentum
        self.stop_epoch_num = self.epoch_num // 3
        if self.stop_epoch_num < 1:
            self.stop_epoch_num = 1

    def build_model(self):
        self.generator = Generator(self.feature_size)
        self.discriminator = Discriminator(self.feature_size, self.data_num)

    def training_prepare(self):
        # decide if the generator training should stop
        self.stop_flag = False

        self.batch_size = min(500, self.data_num)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.optimizer_d = get_optimizer_by_name(self.discriminator, 'sgd',
                                                 lr=self.lr_d,
                                                 sgd_momentum=self.momentum)
        self.optimizer_g = get_optimizer_by_name(self.generator, 'sgd',
                                                 lr=self.lr_g,
                                                 sgd_momentum=self.momentum)

        if self.use_compile:
            self.generator = torch.compile(model=self.generator,
                                           mode=self.compile_mode)
            self.discriminator = torch.compile(model=self.discriminator,
                                               mode=self.compile_mode)
            print('Model compiled.')

    def evaluating_prepare(self):
        self.generator.eval()
        self.discriminator.eval()

        self.generator.train()
        self.discriminator.train()

    def training_forward(self, batch_data):
        batch_data = batch_data.to(self.device)
        batch_data_num = batch_data.size(0)
        self.optimizer_d.zero_grad()

        # Draw samples from the uniform distribution
        noise = torch.rand(batch_data_num, self.feature_size,
                           device=self.device)

        # Train Discriminator
        generated_data = self.generator(noise)

        real_labels = torch.ones(batch_data_num, 1, device=self.device)
        fake_labels = torch.zeros(batch_data_num, 1, device=self.device)

        real_outputs = self.discriminator(batch_data)
        fake_outputs = self.discriminator(generated_data)

        loss_d_real = self.criterion(real_outputs, real_labels)
        loss_d_fake = self.criterion(fake_outputs, fake_labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()

        # Train Generator
        if not self.stop_flag:
            self.optimizer_g.zero_grad()
            trick_labels = torch.ones(batch_data_num, 1, device=self.device)
            loss_g = self.criterion(self.discriminator(noise), trick_labels)
            loss_g.backward()
            self.optimizer_g.step()
        else:
            loss_g = torch.tensor(0.0)

        return loss_d.item(), loss_g.item()

    def epoch_update(self):
        if self.epoch_num >= self.stop_epoch_num:
            self.stop_flag = True

    def evaluating_forward(self, batch_data):
        batch_data = batch_data.to(self.device)
        score = self.discriminator(batch_data).cpu().numpy().ravel()
        return score


class Generator(nn.Module):
    def __init__(self, feature_size):
        super(Generator, self).__init__()
        self.block1 = LinearBlock(feature_size, feature_size,
                                  activation_name='relu', init_type="eye")
        self.block2 = LinearBlock(feature_size, feature_size,
                                  activation_name='relu', init_type="eye")

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, feature_size, data_num):
        super(Discriminator, self).__init__()
        intermidiate_size = math.ceil(math.sqrt(data_num))
        self.block1 = LinearBlock(feature_size, intermidiate_size,
                                  activation_name='relu',
                                  init_type="kaiming_normal",
                                  init_params={'kaiming_mode': 'fan_in',
                                               'kaiming_nonlinearity': 'relu'})
        self.block2 = LinearBlock(intermidiate_size, 1,
                                  activation_name='sigmoid',
                                  init_type="kaiming_normal",
                                  init_params={'kaiming_mode': 'fan_in',
                                               'kaiming_nonlinearity': 'sigmoid'})

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
