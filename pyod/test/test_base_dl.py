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

    def test_fit_with_y_is_ignored(self):
        zero_scores = np.zeros(self.n_train)

        dummy_clf = DummyDetector()
        dummy_clf.fit(self.X_train, self.y_train)
        self.assertEqual(dummy_clf.decision_scores_.all(), zero_scores.all())

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

    def test_save_load_map_location(self):
        # Verify map_location='cpu' updates detector.device and allows
        # inference — exercises the cross-device loading path.
        zero_scores = np.zeros(self.n_train)

        dummy_clf = DummyDetector(verbose=0)
        dummy_clf.fit(self.X_train)
        dummy_clf.save('dummy_clf_cpu.pt')

        loaded = DummyDetector.load('dummy_clf_cpu.pt', map_location='cpu')
        self.assertEqual(loaded.device, torch.device('cpu'))
        self.assertEqual(
            loaded.decision_function(self.X_train).all(),
            zero_scores.all())

        os.remove('dummy_clf_cpu.pt')

    def test_save_load_compiled_model_unwrap(self):
        # torch.compile wraps state-dict keys as '_orig_mod.<name>'.
        # save() must unwrap via _orig_mod before calling state_dict() so that
        # load_state_dict() succeeds on the uncompiled model built by build_model().
        # We simulate the wrapper directly to avoid torch.compile limitations on
        # some platforms (e.g. Windows Inductor).
        zero_scores = np.zeros(self.n_train)

        dummy_clf = DummyDetector(verbose=0)
        dummy_clf.fit(self.X_train)
        orig_model = dummy_clf.model

        class FakeCompiledModule:
            """Mimics OptimizedModule returned by torch.compile."""
            def __init__(self, mod):
                self._orig_mod = mod

            def state_dict(self):
                # compiled modules prefix every key with '_orig_mod.'
                return {'_orig_mod.' + k: v
                        for k, v in self._orig_mod.state_dict().items()}

        dummy_clf.model = FakeCompiledModule(orig_model)
        dummy_clf.save('dummy_clf_compiled.pt')
        dummy_clf.model = orig_model  # restore so teardown is clean

        loaded = DummyDetector.load('dummy_clf_compiled.pt')
        self.assertEqual(
            loaded.decision_function(self.X_train).all(),
            zero_scores.all())

        os.remove('dummy_clf_compiled.pt')

    def test_save_load_base_class_call(self):
        # BaseDeepLearningDetector.load() must return the saved subclass,
        # not try to instantiate the abstract base class directly.
        zero_scores = np.zeros(self.n_train)

        dummy_clf = DummyDetector(verbose=0)
        dummy_clf.fit(self.X_train)
        dummy_clf.save('dummy_clf_base.pt')

        loaded = BaseDeepLearningDetector.load('dummy_clf_base.pt')
        self.assertIsInstance(loaded, DummyDetector)
        self.assertEqual(
            loaded.decision_function(self.X_train).all(),
            zero_scores.all())

        os.remove('dummy_clf_base.pt')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
