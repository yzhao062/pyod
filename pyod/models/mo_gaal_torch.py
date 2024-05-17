import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import check_array
import numpy as np
from .base import BaseDetector
from .gaal_base_torch import create_discriminator, create_generator

class PyODDataset(torch.utils.data.Dataset):
    """Custom Dataset for handling data operations in PyTorch for outlier detection."""
    def __init__(self, X):
        super(PyODDataset, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class MO_GAAL(BaseDetector):
    def __init__(self, k=10, stop_epochs=20, lr_d=0.0005, lr_g=0.0001, momentum=0.9, contamination=0.1):
        super(MO_GAAL, self).__init__(contamination=contamination)
        self.k = k
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.momentum = momentum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = None
        self.generators = []

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)
        n_samples, n_features = X.shape

        self.discriminator = create_discriminator(n_features, n_samples).to(self.device)
        self.generators = [create_generator(n_features).to(self.device) for _ in range(self.k)]

        self.discriminator.apply(self.init_weights)
        for gen in self.generators:
            gen.apply(self.init_weights)

        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        opts_g = [optim.Adam(gen.parameters(), lr=self.lr_g, betas=(0.5, 0.999)) for gen in self.generators]
        criterion = nn.BCELoss()

        dataset = PyODDataset(X)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        num_batches = len(loader)
        for epoch in range(self.stop_epochs * 3):
            self.discriminator.train()
            for index, real_data in enumerate(loader):
                real_data = real_data.to(self.device)
                real_labels = torch.ones(real_data.size(0), 1, device=self.device, dtype=torch.float32)
                fake_labels = torch.zeros(real_data.size(0), 1, device=self.device, dtype=torch.float32)

                self.discriminator.zero_grad()
                real_loss = criterion(self.discriminator(real_data), real_labels)

                fake_data = [gen(torch.randn(real_data.size(0), n_features).to(self.device)) for gen in self.generators]
                fake_losses = [criterion(self.discriminator(f_data.detach()), fake_labels) for f_data in fake_data]
                fake_loss = torch.mean(torch.stack(fake_losses))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                opt_d.step()

                for gen, opt_g in zip(self.generators, opts_g):
                    gen.zero_grad()
                    g_loss = criterion(self.discriminator(gen(torch.randn(real_data.size(0), n_features).to(self.device))), real_labels)
                    g_loss.backward()
                    opt_g.step()

            print(f'Epoch {epoch+1}/{self.stop_epochs * 3}, Loss_D: {d_loss.item()}, Loss_G: {g_loss.item()}')

        self.discriminator.eval()
        with torch.no_grad():
            decision_scores = self.discriminator(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
            self.decision_scores_ = decision_scores.ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        X = check_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.discriminator.eval()
        with torch.no_grad():
            scores = self.discriminator(X_tensor).cpu().numpy()
        return scores

    def predict(self, X, return_confidence=False):
        scores = self.decision_function(X)
        y_pred = (scores > self.threshold_).astype('int').ravel()
        if return_confidence:
            return y_pred, scores.reshape(-1, 1)  # Ensure scores have shape (n_samples, 1)
        return y_pred
