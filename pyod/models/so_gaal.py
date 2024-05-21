import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseDetector



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

class Discriminator(nn.Module):
    def __init__(self, latent_size, data_size):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(latent_size, math.ceil(math.sqrt(data_size)))
        self.layer2 = nn.Linear(math.ceil(math.sqrt(data_size)), 1)
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='sigmoid')

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

class SO_GAAL(BaseDetector):
    def __init__(self, stop_epochs=20, lr_d=0.01, lr_g=0.0001, momentum=0.9, contamination=0.1):
        super(SO_GAAL, self).__init__(contamination=contamination)
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.momentum = momentum

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)
        latent_size = X.shape[1]
        data_size = X.shape[0]
        stop = 0
        epochs = self.stop_epochs * 3
        self.train_history = defaultdict(list)

        self.discriminator = Discriminator(latent_size, data_size)
        self.generator = Generator(latent_size)

        optimizer_d = optim.SGD(self.discriminator.parameters(), lr=self.lr_d, momentum=self.momentum)
        optimizer_g = optim.SGD(self.generator.parameters(), lr=self.lr_g, momentum=self.momentum)
        criterion = nn.BCELoss()

        dataloader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                                batch_size=min(500, data_size),
                                shuffle=True)

        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))

            for data_batch in dataloader:
                data_batch = data_batch[0]
                batch_size = data_batch.size(0)

                # Train Discriminator
                noise = torch.rand(batch_size, latent_size)
                generated_data = self.generator(noise)

                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                outputs_real = self.discriminator(data_batch)
                outputs_fake = self.discriminator(generated_data)

                d_loss_real = criterion(outputs_real, real_labels)
                d_loss_fake = criterion(outputs_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

                self.train_history['discriminator_loss'].append(d_loss.item())

                if stop == 0:
                    # Train Generator
                    trick_labels = torch.ones(batch_size, 1)
                    g_loss = criterion(self.discriminator(self.generator(noise)), trick_labels)

                    optimizer_g.zero_grad()
                    g_loss.backward()
                    optimizer_g.step()

                    self.train_history['generator_loss'].append(g_loss.item())
                else:
                    g_loss = criterion(self.discriminator(self.generator(noise)), trick_labels)
                    self.train_history['generator_loss'].append(g_loss.item())

            if epoch + 1 > self.stop_epochs:
                stop = 1

        self.decision_scores_ = self.discriminator(torch.tensor(X, dtype=torch.float32)).detach().numpy().ravel()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['discriminator'])
        X = check_array(X)
        pred_scores = self.discriminator(torch.tensor(X, dtype=torch.float32)).detach().numpy().ravel()
        return pred_scores
