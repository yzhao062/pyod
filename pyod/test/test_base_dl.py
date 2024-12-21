# -*- coding: utf-8 -*-

import os
import sys
import unittest

import numpy as np
import torch
from torch import nn

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
sys.path.append(os.path.abspath(os.path.dirname("__file__")))

from pyod.models.base_dl import BaseDeepLearningDetector
from pyod.utils.data import generate_data


def loss_function(output, target):
    return torch.mean((output - target) ** 2)


class DummyLoss(nn.Module):
    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean((output - target) ** 2)


class DummyUnchangeModel(nn.Module):
    def __init__(self, feature_size):
        super(DummyUnchangeModel, self).__init__()
        self.layer1 = nn.Linear(feature_size, 2)

    def forward(self, x):
        return self.layer1(x)


class DummyDetector(BaseDeepLearningDetector):
    def __init__(self, contamination=0.1, epoch_num=1, optimizer_name='adam',
                 loss_func=None, criterion=None, criterion_name='mse',
                 verbose=1, preprocessing=True, use_compile=False):
        super(DummyDetector, self).__init__(contamination=contamination,
                                            epoch_num=epoch_num,
                                            optimizer_name=optimizer_name,
                                            loss_func=loss_func,
                                            criterion=criterion,
                                            criterion_name=criterion_name,
                                            verbose=verbose,
                                            preprocessing=preprocessing,
                                            use_compile=use_compile)

    def build_model(self):
        self.model = DummyUnchangeModel(self.feature_size)

    def training_forward(self, batch_data):
        x = batch_data
        x = x.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluating_forward(self, batch_data):
        return np.zeros(batch_data.shape[0])


class DummyDetector2(DummyDetector):
    def __init__(self, contamination=0.1, epoch_num=1, optimizer_name='adam',
                 loss_func=None, criterion=None, criterion_name='mse',
                 verbose=1, preprocessing=True, use_compile=False):
        super(DummyDetector2, self).__init__(contamination=contamination,
                                             epoch_num=epoch_num,
                                             optimizer_name=optimizer_name,
                                             loss_func=loss_func,
                                             criterion=criterion,
                                             criterion_name=criterion_name,
                                             verbose=verbose,
                                             preprocessing=preprocessing,
                                             use_compile=use_compile)

    def build_model(self):
        self.model = DummyUnchangeModel(self.feature_size)

    def training_forward(self, batch_data):
        x = batch_data
        x = x.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, x)
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss.item()


class TestBaseDL(unittest.TestCase):
    def assertHasAttr(self, obj, intended_attr):
        self.assertTrue(hasattr(obj, intended_attr))

    def assertNotHasAttr(self, obj, intended_attr):
        self.assertFalse(hasattr(obj, intended_attr))

    def setUp(self):
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination)

    def test_init(self):
        dummy_clf = DummyDetector()
        dummy_clf.fit(self.X_train)
        self.assertEqual(dummy_clf.contamination, 0.1)
        self.assertIsInstance(dummy_clf.optimizer, torch.optim.Adam)
        self.assertIsInstance(dummy_clf.criterion, nn.MSELoss)

        dummy_clf = DummyDetector(contamination=0.2, optimizer_name='sgd',
                                  loss_func=loss_function, criterion='mae')
        dummy_clf.fit(self.X_train)
        self.assertEqual(dummy_clf.contamination, 0.2)
        self.assertIsInstance(dummy_clf.optimizer, torch.optim.SGD)
        self.assertEqual(dummy_clf.criterion, loss_function)

        dummy_clf = DummyDetector(criterion=DummyLoss())
        self.assertIsInstance(dummy_clf.criterion, DummyLoss)

        dummy_clf = DummyDetector(criterion_name='mae')
        self.assertIsInstance(dummy_clf.criterion, nn.L1Loss)

        self.assertRaises(ValueError, DummyDetector, contamination=0)
        self.assertRaises(ValueError, DummyDetector, contamination=0.51)
        with self.assertRaises(ValueError):
            dummy_clf = DummyDetector(optimizer_name='dummy_optimizer')
            dummy_clf.fit(self.X_train)
        self.assertRaises(ValueError, DummyDetector, loss_func=0)
        self.assertRaises(ValueError, DummyDetector, criterion=0)
        self.assertRaises(ValueError, DummyDetector,
                          criterion_name='dummy_criterion')

    def test_fit_decision_function(self):
        zero_scores = np.zeros(self.n_train)

        dummy_clf = DummyDetector()
        dummy_clf.fit(self.X_train)
        self.assertEqual(dummy_clf.decision_scores_.all(), zero_scores.all())

        dummy_clf = DummyDetector(preprocessing=False)
        dummy_clf.fit(self.X_train)
        self.assertEqual(dummy_clf.decision_scores_.all(), zero_scores.all())

        dummy_clf = DummyDetector(verbose=0)
        dummy_clf.fit(self.X_train)
        self.assertEqual(dummy_clf.decision_scores_.all(), zero_scores.all())

        dummy_clf = DummyDetector(verbose=2)
        dummy_clf.fit(self.X_train)
        self.assertEqual(dummy_clf.decision_scores_.all(), zero_scores.all())

        dummy_clf_2 = DummyDetector2(verbose=2)
        dummy_clf_2.fit(self.X_train)
        self.assertEqual(dummy_clf_2.decision_scores_.all(), zero_scores.all())

        # dummy_clf = DummyDetector(use_compile=True)
        # dummy_clf.fit(self.X_train)
        # self.assertEqual(dummy_clf.decision_scores_.all(), zero_scores.all())

    def test_save_load(self):
        zero_scores = np.zeros(self.n_train)

        dummy_clf = DummyDetector()
        dummy_clf.fit(self.X_train)
        self.assertEqual(dummy_clf.decision_function(self.X_train).all(),
                         zero_scores.all())
        dummy_clf.save('dummy_clf.txt')
        self.assertTrue(os.path.exists('dummy_clf.txt'))

        loaded_dummy_clf = DummyDetector.load('dummy_clf.txt')
        self.assertEqual(
            loaded_dummy_clf.decision_function(self.X_train).all(),
            zero_scores.all())

        os.remove('dummy_clf.txt')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
