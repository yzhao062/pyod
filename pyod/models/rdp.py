"""Neural network models for regression and feature transformation.
This module includes models designed to predict continuous outcomes and transform features using
deep learning techniques.
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.random import sample_without_replacement
from sklearn.neighbors import KDTree

# Constants
MAX_GRAD_NORM = 0.1  # Maximum gradient norm for clipping
LR_GAMMA = 0.1  # Learning rate decay multiplier
LR_DECAY_EPOCHS = 5000  # Epoch interval to decay learning rate
MAX_INT = np.iinfo(np.int32).max
MAX_FLOAT = np.finfo(np.float32).max

# Activation and initialization configuration
cos_activation = False  # Whether to use cosine activation
init_method = 'kaiming'  # Weight initialization method

class RTargetNet(nn.Module):
    """Target network in the RDP model."""
    def __init__(self, in_c, out_c):
        super(RTargetNet, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(inplace=True, negative_slope=0.25) if not cos_activation and init_method != 'rn_orthogonal' else None
        ])
        self._init_weights()

    def forward(self, x):
        x = self.layers(x)
        return torch.cos(x) if cos_activation else x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                if cos_activation:
                    m.weight.data.normal_(std=stdv)
                    m.bias.data.uniform_(0, math.pi) if m.bias is not None else None
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif init_method == 'rn_orthogonal':
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
                elif init_method in ['rn_uniform', 'rn_normal']:
                    m.weight.data.uniform_(-stdv, stdv) if init_method == 'rn_uniform' else m.weight.data.normal_(std=stdv)
                    if m.bias is not None:
                        m.bias.data.uniform_(-stdv, stdv) if init_method == 'rn_uniform' else m.bias.data.normal_(std=stdv)

class RNet(nn.Module):
    """Predictive network in the RDP model, including dropout."""
    def __init__(self, in_c, out_c, dropout_r):
        super(RNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout_r),
            nn.Linear(out_c, out_c)
        )
        self._init_weights()

    def forward(self, x):
        return self.layers(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

class RDP_Model:
    """The RDP (Regression with Dropout Prediction) model class."""
    def __init__(self, in_c, out_c, logfile=None, USE_GPU=False, LR=1e-4, dropout_r=0.2):
        self.r_target_net = RTargetNet(in_c, out_c)
        self.r_net = RNet(in_c, out_c, dropout_r)
        self.USE_GPU = USE_GPU
        self.LR = LR
        self.logfile = logfile

        if USE_GPU:
            self.r_target_net.cuda()
            self.r_net.cuda()

        self.r_net_optim = torch.optim.SGD(self.r_net.parameters(), lr=LR, momentum=0.9)

    def train_model(self, x, epoch):
        self.r_net.train()
        x_random = copy.deepcopy(x)
        np.random.shuffle(x_random)
        x_random = torch.FloatTensor(x_random)
        x = torch.FloatTensor(x)

        if self.USE_GPU:
            x = x.cuda()
            x_random = x_random.cuda()

        r_target = self.r_target_net(x).detach()
        r_pred = self.r_net(x)
        gap_loss = F.mse_loss(r_pred, r_target, reduction='mean')

        r_target_random = self.r_target_net(x_random).detach()
        r_pred_random = self.r_net(x_random)
        consistency_loss = F.mse_loss(r_pred, r_pred_random, reduction='mean')

        loss = gap_loss + consistency_loss

        self.r_net_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.r_net.parameters(), MAX_GRAD_NORM)
        self.r_net_optim.step()

        if epoch % LR_DECAY_EPOCHS == 0 and self.epoch != epoch:
            self.adjust_learning_rate()
        return gap_loss.item(), consistency_loss.item()

    def adjust_learning_rate(self):
        self.LR *= LR_GAMMA
        for param_group in self.r_net_optim.param_groups:
            param_group['lr'] = self.LR

    def evaluate_model(self, x):
        """Evaluate the model on a given batch of data, return evaluation metrics."""
        self.r_net.eval()
        x = torch.FloatTensor(x)
        if self.USE_GPU:
            x = x.cuda()

        r_target = self.r_target_net(x)
        r_pred = self.r_net(x)
        gap_loss = F.mse_loss(r_pred, r_target, reduction='mean')

        return {
            'gap_loss': gap_loss.item()
        }
