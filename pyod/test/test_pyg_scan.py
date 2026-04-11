# -*- coding: utf-8 -*-
"""Tests for SCAN graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_scan import SCAN
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestSCAN(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=200, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = SCAN(epsilon=0.5, mu=2, contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 200
        assert len(clf.labels_) == 200

    def test_fit_numpy(self):
        clf = SCAN(epsilon=0.5, mu=2, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 200

    def test_scores_nonnegative(self):
        clf = SCAN(contamination=0.1)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = SCAN()
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = SCAN()
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_structure_only(self):
        """SCAN works without node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=200)
        clf = SCAN(contamination=0.1)
        clf.fit(data_no_feat)
        assert len(clf.decision_scores_) == 200

    def test_empty_graph(self):
        """Isolated nodes with no edges all score 1.0."""
        data_empty = Data(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            num_nodes=10)
        clf = SCAN(contamination=0.1)
        clf.fit(data_empty)
        assert len(clf.decision_scores_) == 10
        assert np.all(clf.decision_scores_ == 1.0)

    def test_epsilon_mu_affect_scores(self):
        """Different epsilon/mu values produce different scores."""
        clf1 = SCAN(epsilon=0.3, mu=1, contamination=0.1)
        clf1.fit(self.data)
        clf2 = SCAN(epsilon=0.9, mu=5, contamination=0.1)
        clf2.fit(self.data)
        assert not np.allclose(
            clf1.decision_scores_, clf2.decision_scores_)


if __name__ == '__main__':
    unittest.main()
