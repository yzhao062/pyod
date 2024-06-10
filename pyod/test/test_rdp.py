# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import torch

# Temporary solution for relative imports in case pyod is not installed
# If pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.rdp import RDP_Model
from pyod.utils.data import generate_data


class TestRDP_Model(unittest.TestCase):
    """
    Unit tests for the RDP_Model.
    The test is designed for running models only, without any performance check.
    """

    def setUp(self):
        self.n_train = 1000
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1

    def test_initialization(self):
        """Test the initialization and setup of the RDP_Model."""
        model = RDP_Model(in_c=self.n_features, out_c=self.n_features, dropout_r=0.2)
        self.assertIsNotNone(model)

    def test_train_model(self):
        """Test training of the RDP_Model."""
        model = RDP_Model(in_c=self.n_features, out_c=self.n_features, dropout_r=0.2)
        X_train, _ = generate_data(n_train=self.n_train, n_features=self.n_features, contamination=self.contamination)
        model.train_model(torch.FloatTensor(X_train), epoch=10)
        self.assertTrue(True)  # Simply check that it completes

    def test_evaluate_model(self):
        """Test evaluation of the RDP_Model."""
        model = RDP_Model(in_c=self.n_features, out_c=self.n_features, dropout_r=0.2)
        X_test, _ = generate_data(n_train=self.n_test, n_features=self.n_features, contamination=self.contamination)
        results = model.evaluate_model(torch.FloatTensor(X_test))
        self.assertIn('gap_loss', results)  # Check if results contain expected output

if __name__ == '__main__':
    unittest.main()
