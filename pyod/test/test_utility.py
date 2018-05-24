import os, sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_less_equal
import numpy as np
from pyod.utils.load_data import generate_data


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


if __name__ == '__main__':
    unittest.main()
