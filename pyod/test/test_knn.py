import unittest
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.metrics import roc_auc_score

import sys

# temporary solution for relative imports
sys.path.append("..")
from pyod.models.knn import Knn
from pyod.utils.load_data import generate_data


class TestKnn(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train, _, self.X_test, self.y_test, _ = generate_data(
            n_train=100, n_test=50, contamination=0.05)

    def test_knn(self):
        clf = Knn(contamination=0.05)
        clf.fit(self.X_train)
        assert_equal(len(clf.decision_scores), self.X_train.shape[0])

        pred_scores = clf.decision_function(self.X_test)
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])
        assert_equal(clf.predict(self.X_test).shape[0],
                     self.X_test.shape[0])
        assert_greater(roc_auc_score(self.y_test, pred_scores), 0.5)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
