import unittest
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater
from sklearn.metrics import roc_auc_score

import sys

# temporary solution for relative imports
sys.path.append("..")
from pyod.utils.utility import precision_n_scores
from pyod.data.load_data import generate_data


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train, _, self.X_test, self.y_test, _ = generate_data(
            n_train=100, n_test=50, contamination=0.05)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
