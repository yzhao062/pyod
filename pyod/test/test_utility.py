'''Testing utilities.'''
import os, sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import unittest
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises
from sklearn.metrics import precision_score

from pyod.utils.load_data import generate_data
from pyod.utils.utility import check_parameter_range
from pyod.utils.utility import standardizer
from pyod.utils.utility import get_label_n
from pyod.utils.utility import precision_n_scores


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.n_train = 1000
        self.n_test = 500
        self.contamination = 0.1
        pass

    def test_data_generate(self):
        X_train, y_train, _, X_test, y_test, _ = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination)

        assert_equal(y_train.shape[0], X_train.shape[0])
        assert_equal(y_test.shape[0], X_test.shape[0])

        assert_less_equal(self.n_train - X_train.shape[0], 1)
        assert_equal(X_train.shape[1], 2)

        assert_less_equal(self.n_test - X_test.shape[0], 1)
        assert_equal(X_test.shape[1], 2)

        out_perc = np.sum(y_train) / self.n_train
        assert_allclose(self.contamination, out_perc, atol=0.01)

        out_perc = np.sum(y_test) / self.n_test
        assert_allclose(self.contamination, out_perc, atol=0.01)

    def tearDown(self):
        pass


class TestParameters(unittest.TestCase):
    def setUp(self):
        pass

    def test_check_para_range(self):
        with assert_raises(ValueError):
            check_parameter_range(2)
        with assert_raises(ValueError):
            check_parameter_range(2, 100, 99)
        with assert_raises(ValueError):
            check_parameter_range(2, 100, 100)
        with assert_raises(TypeError):
            check_parameter_range('f', 3, 10)

    def tearDown(self):
        pass


class TestScaler(unittest.TestCase):

    def setUp(self):
        self.X_train = np.random.rand(500, 5)
        self.X_test = np.random.rand(50, 5)

    def test_normalization(self):
        norm_X_train, norm_X_test = standardizer(self.X_train, self.X_train)
        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        assert_allclose(norm_X_test.mean(), 0, atol=0.05)
        assert_allclose(norm_X_test.std(), 1, atol=0.05)

    def tearDown(self):
        pass


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        self.y_pred = [0.1, 0.2, 0.2, 0.8, 0.2, 0.5, 0.7, 0.9, 1, 0.3]
        self.manual_y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
        self.outlier_perc = 0.3

    def test_precision_n_scores(self):
        assert_equal(precision_score(self.y, self.manual_y_pred),
                     precision_n_scores(self.y, self.y_pred))

    def test_get_label_n(self):
        assert_allclose(self.manual_y_pred,
                        get_label_n(self.y, self.y_pred))

    def test_get_label_n_equal_3(self):
        manual_y_pred = [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
        assert_allclose(manual_y_pred,
                        get_label_n(self.y, self.y_pred, n=3))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
