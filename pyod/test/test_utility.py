import unittest

import sys

# temporary solution for relative imports
sys.path.append("..")
from pyod.utils.load_data import generate_data


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train, _, self.X_test, self.y_test, _ = generate_data(
            n_train=100, n_test=50, contamination=0.05)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
