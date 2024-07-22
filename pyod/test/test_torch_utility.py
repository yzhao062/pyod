# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
sys.path.append(os.path.abspath(os.path.dirname("__file__")))

from pyod.utils.torch_utility import *


class TestBaseDL(unittest.TestCase):
    def setUp(self):
        # create a dummy dataset
        self.X_train = torch.ones(2, 2)
        self.y_train = torch.ones(2)
        self.X_test = torch.ones(1, 2)
        self.y_test = torch.ones(1)
        self.mean = torch.mean(self.X_train, dim=0)
        self.std = torch.std(self.X_train, dim=0)

    def test_torch_dataset(self):
        train_dataset = TorchDataset(X=self.X_train, y=self.y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=2,
                                                   shuffle=True)
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(train_loader), 1)
        for data, target in train_loader:
            self.assertTrue(torch.equal(data, torch.ones(2, 2)))
            self.assertTrue(torch.equal(target, torch.ones(2)))

        train_dataset = TorchDataset(X=self.X_train, mean=self.mean,
                                     std=self.std)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=2,
                                                   shuffle=True)
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(train_loader), 1)
        for data in train_loader:
            self.assertTrue(torch.equal(data, torch.zeros(2, 2)))

    def test_linear_block(self):
        train_dataset = TorchDataset(X=self.X_train, mean=self.mean,
                                     std=self.std)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=2,
                                                   shuffle=True)
        dummy_block = LinearBlock(in_features=2, out_features=1,
                                  batch_norm=True, dropout_rate=0.2)

        for data in train_loader:
            output = dummy_block(data)
            self.assertTrue(torch.equal(output, torch.zeros(2, 1)))

    def test_get_activation_by_name(self):
        # test relu activation
        dummy_relu = get_activation_by_name('relu')
        self.assertIsInstance(dummy_relu, nn.ReLU)
        self.assertEqual(dummy_relu.inplace, False)
        self.assertTrue(
            torch.equal(
                dummy_relu(torch.tensor([-1.0, 0.0, 1.0])),
                torch.tensor([0.0, 0.0, 1.0])
            )
        )

        # test leaky relu activation
        dummy_elu = get_activation_by_name('elu', elu_alpha=1.0)
        self.assertIsInstance(dummy_elu, nn.ELU)
        self.assertEqual(dummy_elu.inplace, False)
        self.assertEqual(dummy_elu.alpha, 1.0)
        self.assertTrue(
            torch.equal(
                dummy_elu(
                    torch.tensor([torch.log(torch.tensor(0.5)), 0.0, 1.0])),
                torch.tensor([-0.5, 0.0, 1.0])
            )
        )

        # test leaky relu activation
        dummy_leaky_relu = get_activation_by_name('leaky_relu',
                                                  leaky_relu_negative_slope=0.1)
        self.assertIsInstance(dummy_leaky_relu, nn.LeakyReLU)
        self.assertEqual(dummy_leaky_relu.inplace, False)
        self.assertEqual(dummy_leaky_relu.negative_slope, 0.1)
        self.assertTrue(
            torch.equal(
                dummy_leaky_relu(torch.tensor([-1.0, 0.0, 1.0])),
                torch.tensor([-0.1, 0.0, 1.0])
            )
        )

        # test sigmoid activation
        dummy_sigmoid = get_activation_by_name('sigmoid')
        self.assertIsInstance(dummy_sigmoid, nn.Sigmoid)
        self.assertTrue(
            torch.equal(
                dummy_sigmoid(torch.tensor([torch.log(torch.tensor(0.25)), 0.0,
                                            torch.log(torch.tensor(4.0))])),
                torch.tensor([0.2, 0.5, 0.8])
            )
        )

        # test softmax activation
        dummy_softmax = get_activation_by_name('softmax', softmax_dim=1)
        self.assertIsInstance(dummy_softmax, nn.Softmax)
        self.assertEqual(dummy_softmax.dim, 1)
        self.assertTrue(
            torch.equal(
                dummy_softmax(torch.tensor(
                    [[0.0, 0.0, torch.log(torch.tensor(2.0))],
                     [0.0, torch.log(torch.tensor(2.0)), 0.0]])),
                torch.tensor([[0.25, 0.25, 0.5], [0.25, 0.5, 0.25]])
            )
        )

        # test softplus activation
        dummy_softplus = get_activation_by_name('softplus',
                                                softplus_beta=1.0,
                                                softplus_threshold=20.0)
        self.assertIsInstance(dummy_softplus, nn.Softplus)
        self.assertEqual(dummy_softplus.beta, 1.0)
        self.assertEqual(dummy_softplus.threshold, 20.0)
        self.assertTrue(
            torch.equal(
                dummy_softplus(torch.tensor([torch.log(torch.tensor(np.e - 1)),
                                             torch.log(
                                                 torch.tensor(np.e ** 2 - 1)),
                                             torch.log(torch.tensor(
                                                 np.e ** 3 - 1))])),
                torch.tensor([1.0, 2.0, 3.0])
            )
        )

        # test tanh activation
        dummy_tanh = get_activation_by_name('tanh')
        self.assertIsInstance(dummy_tanh, nn.Tanh)
        self.assertTrue(
            torch.equal(
                dummy_tanh(torch.tensor([torch.log(torch.tensor(0.5)), 0.0,
                                         torch.log(torch.tensor(2.0))])),
                torch.tensor([-0.6, 0.0, 0.6])
            )
        )

        # test invalid activation
        self.assertRaises(ValueError, get_activation_by_name, name='random')

    def test_get_optimizer_by_name(self):
        # define a dummy model
        dummy_model = nn.Linear(2, 1)

        # test adam optimizer
        dummy_optimizer = get_optimizer_by_name(model=dummy_model,
                                                name='adam',
                                                lr=0.1,
                                                weight_decay=0.01,
                                                adam_eps=1e-8)
        self.assertIsInstance(dummy_optimizer, torch.optim.Adam)
        self.assertEqual(dummy_optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual(dummy_optimizer.param_groups[0]['weight_decay'], 0.01)
        self.assertEqual(dummy_optimizer.param_groups[0]['eps'], 1e-8)

        # test sgd optimizer
        dummy_optimizer = get_optimizer_by_name(model=dummy_model,
                                                name='sgd',
                                                lr=0.1,
                                                weight_decay=0.01,
                                                sgd_momentum=0.9,
                                                sgd_nesterov=False)
        self.assertIsInstance(dummy_optimizer, torch.optim.SGD)
        self.assertEqual(dummy_optimizer.param_groups[0]['lr'], 0.1)
        self.assertEqual(dummy_optimizer.param_groups[0]['momentum'], 0.9)
        self.assertEqual(dummy_optimizer.param_groups[0]['weight_decay'], 0.01)

        # test invalid optimizer
        self.assertRaises(ValueError, get_optimizer_by_name, model=dummy_model,
                          name='random')

    def test_get_criterion_by_name(self):
        # test mse loss with reduction mean
        dummy_criterion = get_criterion_by_name(name='mse', reduction='mean')
        self.assertIsInstance(dummy_criterion, nn.MSELoss)
        self.assertTrue(
            torch.equal(
                dummy_criterion(torch.tensor([3.0, 3.0]),
                                torch.tensor([0.0, 0.0])),
                torch.tensor(9.0)
            )
        )

        # test mse loss with reduction sum
        dummy_criterion = get_criterion_by_name(name='mse', reduction='sum')
        self.assertIsInstance(dummy_criterion, nn.MSELoss)
        self.assertTrue(
            torch.equal(
                dummy_criterion(torch.tensor([3.0, 3.0]),
                                torch.tensor([0.0, 0.0])),
                torch.tensor(18.0)
            )
        )

        # test mse loss with reduction none
        dummy_criterion = get_criterion_by_name(name='mse', reduction='none')
        self.assertIsInstance(dummy_criterion, nn.MSELoss)
        self.assertTrue(
            torch.equal(
                dummy_criterion(torch.tensor([3.0, 3.0]),
                                torch.tensor([0.0, 0.0])),
                torch.tensor([9.0, 9.0])
            )
        )

        # test mae(l1) loss with reduction none
        dummy_criterion = get_criterion_by_name(name='mae', reduction='none')
        self.assertIsInstance(dummy_criterion, nn.L1Loss)
        self.assertTrue(
            torch.equal(
                dummy_criterion(torch.tensor([3.0, 3.0]),
                                torch.tensor([0.0, 0.0])),
                torch.tensor([3.0, 3.0])
            )
        )

        # test bce loss (for binary classification) with reduction none
        dummy_criterion = get_criterion_by_name(name='bce', reduction='none')
        self.assertIsInstance(dummy_criterion, nn.BCELoss)
        self.assertTrue(
            torch.equal(
                dummy_criterion(torch.tensor([1 / np.e, 1 - 1 / np.e]),
                                torch.tensor([1.0, 0.0])),
                torch.tensor([1.0, 1.0])
            )
        )

        # test invalid criterion
        self.assertRaises(ValueError, get_criterion_by_name, name='random')

    def test_init_weights(self):
        # define a dummy layer
        dummy_layer = nn.Linear(2, 1)

        # For the following initializers, 
        # we only test if the function can be called without error 
        # since the actual initialization is random and cannot be tested.
        for name in ['uniform', 'normal', 'xavier_uniform',
                     'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                     'trunc_normal', 'orthogonal']:
            init_weights(layer=dummy_layer, name=name)
        init_weights(layer=dummy_layer, name='sparse', sparse_sparsity=0.1)

        # test constant initializer
        init_weights(layer=dummy_layer, name='constant', constant_val=0.1)
        self.assertTrue(
            torch.equal(dummy_layer.weight, torch.tensor([[0.1, 0.1]])))

        # test ones initializer
        init_weights(layer=dummy_layer, name='ones')
        self.assertTrue(
            torch.equal(dummy_layer.weight, torch.tensor([[1.0, 1.0]])))

        # test zeros initializer
        init_weights(layer=dummy_layer, name='zeros')
        self.assertTrue(
            torch.equal(dummy_layer.weight, torch.tensor([[0.0, 0.0]])))

        # test eye initializer
        init_weights(layer=dummy_layer, name='eye')
        self.assertTrue(
            torch.equal(dummy_layer.weight, torch.tensor([[1.0, 0.0]])))

        # test invalid initializer
        self.assertRaises(ValueError, init_weights, layer=dummy_layer,
                          name='random')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
