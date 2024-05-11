import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseDetector
from .gaal_base_torch import create_discriminator, create_generator

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader"""

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std

        return torch.from_numpy(sample), idx

class MO_GAAL(BaseDetector):
    def __init__(self, k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001, momentum=0.9, contamination=0.1, hidden_size=64):
        super(MO_GAAL, self).__init__(contamination=contamination)
        self.k = k
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.momentum = momentum
        self.hidden_size = hidden_size

    def fit(self, X, y=None):
        X = check_array(X)
        X = torch.tensor(X, dtype=torch.float32)
        self._set_n_classes(y)
        self.train_history = defaultdict(list)
        names = locals()
        epochs = self.stop_epochs * 3
        latent_size = X.shape[1]
        data_size = X.shape[0]

        # Create discriminator
        self.discriminator = create_discriminator(latent_size, data_size)
        discriminator_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr_d, momentum=self.momentum)
        criterion = nn.BCELoss()

        # Create k combine models
        for i in range(self.k):
            names['sub_generator' + str(i)] = create_generator(latent_size)
            self.discriminator.trainable = False
            names['combine_model' + str(i)] = nn.Sequential(names['sub_generator' + str(i)], self.discriminator)
            names['combine_model_optimizer' + str(i)] = optim.SGD(names['combine_model' + str(i)].parameters(), lr=self.lr_g, momentum=self.momentum)

        # Start iteration
        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)

            for index in range(num_batches):
                print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                # Generate noise
                noise_size = batch_size
                noise = torch.FloatTensor(np.random.uniform(0, 1, (int(noise_size), latent_size)))

                # Get training data
                data_batch = torch.FloatTensor(X[index * batch_size: (index + 1) * batch_size])

                # Generate potential outliers
                generated_data = []
                block = ((1 + self.k) * self.k) // 2
                for i in range(self.k):
                    if i != (self.k - 1):
                        noise_start = int((((self.k + (self.k - i + 1)) * i) / 2) * (noise_size // block))
                        noise_end = int((((self.k + (self.k - i)) * (i + 1)) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start:noise_end, :]
                    else:
                        noise_start = int((((self.k + (self.k - i + 1)) * i) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start:noise_size, :]

                    names['generated_data' + str(i)] = names['sub_generator' + str(i)](names['noise' + str(i)])
                    generated_data.append(names['generated_data' + str(i)])

                # Concatenate real data to generated data
                x_list = [data_batch] + [names['generated_data' + str(i)] for i in range(self.k)]
                x = torch.cat(x_list, dim=0)
                y = torch.FloatTensor([1] * batch_size + [0] * int(noise_size))
                y = y.view(-1, 1)

                # Train discriminator
                discriminator_optimizer.zero_grad()
                discriminator_outputs = self.discriminator(x)
                discriminator_outputs = discriminator_outputs.view(-1, 1)
                discriminator_loss = criterion(discriminator_outputs, y)
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # Record discriminator loss
                self.train_history['discriminator_loss'].append(discriminator_loss.item())

                # Get the target value of sub-generator
                pred_scores = self.discriminator(x)

                for i in range(self.k):
                    names['T' + str(i)] = np.percentile(pred_scores.detach().numpy(), i / self.k * 100)
                    names['trick' + str(i)] = np.array([float(names['T' + str(i)])] * noise_size)

                # Train generator
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                if epoch + 1 > self.stop_epochs:
                    for i in range(self.k):
                        noise_tensor = torch.tensor(noise, dtype=torch.float32)
                        trick_tensor = torch.tensor(names['trick' + str(i)], dtype=torch.float32).view(-1,1)

                        loss = criterion(names['combine_model' + str(i)](noise_tensor), trick_tensor)

                        names['combine_model_optimizer' + str(i)].zero_grad()
                        loss.backward()
                        names['combine_model_optimizer' + str(i)].step()

                        names['sub_generator' + str(i) + '_loss'] = loss.item()
                        self.train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])

                generator_loss = sum(names.get('sub_generator' + str(i) + '_loss', 0) for i in range(self.k)) / self.k
                self.train_history['generator_loss'].append(generator_loss)

        self.decision_scores_ = self.discriminator(X).detach().numpy().ravel()
        self._process_decision_scores()
        return self


    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['discriminator'])
        X = torch.tensor(X, dtype=torch.float32)
        pred_scores = self.discriminator(X).detach().numpy().ravel()
        return pred_scores
