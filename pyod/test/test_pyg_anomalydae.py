# -*- coding: utf-8 -*-
"""Tests for AnomalyDAE graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_anomalydae import AnomalyDAE
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestAnomalyDAE(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=100, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5,
                         contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 100

    def test_fit_numpy(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 100

    def test_scores_nonnegative(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """AnomalyDAE requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=100)
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)


if __name__ == '__main__':
    unittest.main()
