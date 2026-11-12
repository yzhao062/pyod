# -*- coding: utf-8 -*-

import io
import os
import pickle
import shutil
import sys
import tempfile
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

    def test_fit_with_fewer_samples_than_batch_size(self):
        """A dataset smaller than one batch must still train.

        The training loader used drop_last=True, so a dataset shorter than
        batch_size produced no batches: train() iterated zero times and then
        read a `loss` that was never assigned, raising UnboundLocalError.
        """
        X_small = self.X_train[:5]
        self.assertLess(len(X_small), DummyDetector().batch_size)

        for dummy_clf in (DummyDetector(verbose=0), DummyDetector2(verbose=0)):
            dummy_clf.fit(X_small)
            self.assertHasAttr(dummy_clf, 'decision_scores_')
            self.assertEqual(len(dummy_clf.decision_scores_), len(X_small))

    def test_fit_returns_self(self):
        # BaseDetector.fit documents "Returns: self : object", but the deep
        # learning base returned None, so `clf.fit(X).predict(X)` raised
        # AttributeError for every detector built on it.
        dummy_clf = DummyDetector()
        returned = dummy_clf.fit(self.X_train)

        self.assertIs(returned, dummy_clf)
        # the chained form users expect from an sklearn-style estimator
        labels = DummyDetector().fit(self.X_train).predict(self.X_test)
        self.assertEqual(labels.shape, (self.n_test,))

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

        loaded_dummy_clf = DummyDetector.load('dummy_clf.txt', trusted=True)
        self.assertEqual(
            loaded_dummy_clf.decision_function(self.X_train).all(),
            zero_scores.all())

        os.remove('dummy_clf.txt')

    def test_save_load_map_location(self):
        # Verify map_location string/device and dict forms all update
        # detector.device to match the remapped weights.
        zero_scores = np.zeros(self.n_train)

        dummy_clf = DummyDetector(verbose=0)
        dummy_clf.fit(self.X_train)
        dummy_clf.save('dummy_clf_cpu.pt')

        loaded = DummyDetector.load('dummy_clf_cpu.pt', map_location='cpu', trusted=True)
        self.assertEqual(loaded.device, torch.device('cpu'))
        self.assertEqual(
            loaded.decision_function(self.X_train).all(),
            zero_scores.all())

        # Dict map_location: tensors are remapped by torch.load, but the old
        # code never updated detector.device for non-str/device forms.
        # Verify detector.device is inferred from the loaded weight tensors.
        loaded_dict = DummyDetector.load(
            'dummy_clf_cpu.pt', map_location={'cpu': 'cpu'}, trusted=True)
        self.assertEqual(loaded_dict.device, torch.device('cpu'))
        self.assertEqual(
            loaded_dict.decision_function(self.X_train).all(),
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

        loaded = DummyDetector.load('dummy_clf_compiled.pt', trusted=True)
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

        loaded = BaseDeepLearningDetector.load('dummy_clf_base.pt', trusted=True)
        self.assertIsInstance(loaded, DummyDetector)
        self.assertEqual(
            loaded.decision_function(self.X_train).all(),
            zero_scores.all())

        os.remove('dummy_clf_base.pt')

    def test_load_rejects_without_trusted(self):
        # load() must refuse to deserialize before torch.load is called
        # when trusted=False (the default), so a malicious pickle cannot
        # execute code before any validation.
        dummy_clf = DummyDetector(verbose=0)
        dummy_clf.fit(self.X_train)
        dummy_clf.save('dummy_clf_untrusted.pt')

        with self.assertRaises(ValueError) as ctx:
            DummyDetector.load('dummy_clf_untrusted.pt')
        self.assertIn('trusted=True', str(ctx.exception))

        os.remove('dummy_clf_untrusted.pt')

    def tearDown(self):
        pass


class TestBaseDLSaveLoadRealDetectors(unittest.TestCase):
    """Save/load round trips on real subclasses rather than a dummy detector
    that returns zeros, so the assertions show that learned scores survive.
    """

    def setUp(self):
        self.X_train, self.X_test, _, _ = generate_data(
            n_train=200, n_test=50, n_features=5, contamination=0.1,
            random_state=42)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmpdir, name)

    def _fit_autoencoder(self):
        from pyod.models.auto_encoder import AutoEncoder
        clf = AutoEncoder(epoch_num=3, hidden_neuron_list=[8, 4],
                          device='cpu', verbose=0, random_state=0)
        clf.fit(self.X_train)
        return clf

    def _fit_so_gaal(self):
        from pyod.models.so_gaal_new import SO_GAAL
        clf = SO_GAAL(epoch_num=3, device='cpu', verbose=0, random_state=0)
        clf.fit(self.X_train)
        return clf

    def test_autoencoder_scores_survive_save_load(self):
        clf = self._fit_autoencoder()
        expected = clf.decision_function(self.X_test)
        path = self._path('ae.pt')
        clf.save(path)

        for map_location in (None, 'cpu', torch.device('cpu'),
                             {'cpu': 'cpu'}, lambda storage, loc: storage):
            loaded = BaseDeepLearningDetector.load(
                path, map_location=map_location, trusted=True)
            self.assertIsInstance(loaded, type(clf))
            self.assertEqual(loaded.device, torch.device('cpu'))
            self.assertFalse(loaded.model.training)
            np.testing.assert_allclose(
                loaded.decision_function(self.X_test), expected, rtol=1e-6)
            np.testing.assert_allclose(
                loaded.decision_scores_, clf.decision_scores_)
            np.testing.assert_array_equal(
                loaded.predict(self.X_test), clf.predict(self.X_test))

    def test_so_gaal_device_inferred_from_non_model_modules(self):
        # SO_GAAL keeps its networks in generator/discriminator, not
        # self.model, so there is no model state dict to infer the device
        # from.  Simulate a CUDA-trained artifact: the stored device attribute
        # is stale and must be corrected from the loaded modules for every
        # map_location form.
        clf = self._fit_so_gaal()
        self.assertIsNone(getattr(clf, 'model', None))
        expected = clf.decision_function(self.X_test)
        clf.device = torch.device('cuda:0')
        path = self._path('so_gaal.pt')
        clf.save(path)

        for map_location in ('cpu', torch.device('cpu'),
                             {'cuda:0': 'cpu'},
                             lambda storage, loc: storage):
            loaded = BaseDeepLearningDetector.load(
                path, map_location=map_location, trusted=True)
            self.assertEqual(loaded.device, torch.device('cpu'))
            for module in (loaded.generator, loaded.discriminator):
                self.assertEqual(next(module.parameters()).device,
                                 torch.device('cpu'))
            np.testing.assert_allclose(
                loaded.decision_function(self.X_test), expected, rtol=1e-6)

    def test_dict_map_location_without_tensors_maps_saved_device(self):
        # With no weights to inspect, a dict map_location is applied to the
        # saved device attribute so detector.device is still not stale.
        clf = self._fit_autoencoder()
        clf.model = None
        clf.device = torch.device('cuda:0')
        path = self._path('no_weights.pt')
        clf.save(path)
        loaded = BaseDeepLearningDetector.load(
            path, map_location={'cuda:0': 'cpu'}, trusted=True)
        self.assertEqual(loaded.device, torch.device('cpu'))

    def test_legacy_pickle_artifact_still_loads(self):
        clf = self._fit_autoencoder()
        expected = clf.decision_function(self.X_test)
        path = self._path('legacy.pkl')
        with open(path, 'wb') as f:
            pickle.dump(clf, f)
        loaded = type(clf).load(path, map_location='cpu', trusted=True)
        np.testing.assert_allclose(
            loaded.decision_function(self.X_test), expected, rtol=1e-6)

    def test_file_like_objects_supported(self):
        clf = self._fit_autoencoder()
        expected = clf.decision_function(self.X_test)

        buf = io.BytesIO()
        clf.save(buf)
        buf.seek(0)
        loaded = type(clf).load(buf, trusted=True)
        np.testing.assert_allclose(
            loaded.decision_function(self.X_test), expected, rtol=1e-6)

        legacy = io.BytesIO(pickle.dumps(clf))
        loaded = type(clf).load(legacy, trusted=True)
        np.testing.assert_allclose(
            loaded.decision_function(self.X_test), expected, rtol=1e-6)

    def test_torch_load_error_not_masked_as_pickle_error(self):
        # A truncated torch archive used to fall through the catch-all
        # fallback into pickle.load, which reported a misleading
        # UnpicklingError.  It must now surface torch.load's own error.
        clf = self._fit_autoencoder()
        path = self._path('truncated.pt')
        clf.save(path)
        with open(path, 'rb') as f:
            data = f.read()
        with open(path, 'wb') as f:
            f.write(data[:len(data) // 2])
        with self.assertRaises(Exception) as ctx:
            type(clf).load(path, trusted=True)
        # The exact type depends on the torch version (RuntimeError or
        # OSError); what matters is that it is not the pickle fallback's.
        self.assertNotIsInstance(ctx.exception, pickle.UnpicklingError)
        self.assertNotIn('invalid load key', str(ctx.exception))

    def test_invalid_map_location_error_not_masked(self):
        clf = self._fit_autoencoder()
        path = self._path('ae_bad_map.pt')
        clf.save(path)
        with self.assertRaises(Exception) as ctx:
            type(clf).load(path, map_location='not_a_device', trusted=True)
        self.assertNotIsInstance(ctx.exception, pickle.UnpicklingError)


if __name__ == '__main__':
    unittest.main()
