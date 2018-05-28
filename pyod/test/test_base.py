import os, sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.metrics import roc_auc_score

from pyod.models.base import BaseDetector
from pyod.utils.load_data import generate_data


class TestBASE(unittest.TestCase):
    def setUp(self):
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.6
        self.X_train, self.y_train, _, self.X_test, self.y_test, _ = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination)

    def test_init(self):
        """
        Test base class initialization

        :return:
        """

        class dummy(BaseDetector):
            def __init__(self, contamination=0.1):
                super().__init__(contamination=contamination)

            def decision_function(self, X):
                pass

            def fit(self, X):
                pass

        self.dummy_clf = dummy()
        assert_equal(self.dummy_clf.contamination, 0.1)

        self.dummy_clf = dummy(contamination=0.2)
        assert_equal(self.dummy_clf.contamination, 0.2)

    def test_fit(self):
        class dummy(BaseDetector):
            def __init__(self, contamination=0.1):
                super().__init__(contamination=contamination)

            def decision_function(self, X):
                pass

            def fit(self, X):
                return X

        self.dummy_clf = dummy()

        assert_equal(self.dummy_clf.fit(0), 0)

    def test_fit_predict(self):
        #TODO: add more testcases
        class dummy(BaseDetector):
            def __init__(self, contamination=0.1):
                super().__init__(contamination=contamination)

            def decision_function(self, X):
                pass

            def fit(self, X):
                self.y_pred = X

        self.dummy_clf = dummy()

        assert_equal(self.dummy_clf.fit_predict(0), 0)

    def test_predict_proba(self):
        # TODO: create uniform testcases
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
