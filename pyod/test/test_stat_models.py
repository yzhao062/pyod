"""Testing utilities."""
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
from pyod.utils.stat_models import wpearsonr


class TestStatModels(unittest.TestCase):
    def setUp(self):
        self.n_train = 1000
        self.n_test = 500
        self.contamination = 0.1

        self.a = [1, 2, 3, 2, 3, 1, 0, 5]
        self.b = [1, 2, 1, 2, 2, 1, 0, 2]
        self.w = [2, 2, 1, 2, 4, 1, 0, 2]

    def test_wpearsonr(self):
        # TODO: if unweight version changes, wp[0] format should be changed
        wp = wpearsonr(self.a, self.b)
        assert_allclose(wp[0], 0.6956083, atol=0.01)

        wp = wpearsonr(self.a, self.b, w=self.w)
        assert_allclose(wp, 0.5477226, atol=0.01)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
