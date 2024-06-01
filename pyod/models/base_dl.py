# -*- coding: utf-8 -*-
"""Base class for deep learning models
"""
# Author: Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import time
from abc import abstractmethod
from inspect import isfunction

import numpy as np
import tensorflow
import torch
import tqdm
from sklearn.utils import check_array

from .base import BaseDetector
from ..utils.torch_utility import TorchDataset, \
	get_optimizer_by_name, get_criterion_by_name


# Old function, deprecat this in the future
def _get_tensorflow_version():  # pragma: no cover
	""" Utility function to decide the version of tensorflow, which will
	affect how to import keras models.

	Returns
	-------
	tensorflow version : int

	"""

	tf_version = str(tensorflow.__version__)
	if int(tf_version.split(".")[0]) != 1 and int(
			tf_version.split(".")[0]) != 2:
		raise ValueError("tensorflow version error")

	return int(tf_version.split(".")[0]) * 100 + int(tf_version.split(".")[1])


class BaseDeepLearningDetector(BaseDetector):
	"""
	Abstract class for all deep learning models.

	Parameters
	----------
	contamination : float in (0., 0.5), optional (default=0.1)
		The amount of contamination of the data set,
		i.e. the proportion of outliers in the data set.
		Used when fitting to define the threshold on the decision function.

	preprocessing : bool, optional (default=True)
		If True, apply the preprocessing step before training the model.

	lr : float, optional (default=1e-3)
		The learning rate for the optimizer.

	epoch_num : int, optional (default=100)
		The number of epochs to train the model.

	batch_size : int, optional (default=32)
		The batch size for training the model.

	optimizer : str, optional (default='adam')
		The optimizer used to train the model.

	loss_func : str, optional (default=None)
		The loss function used to train the model.

	criterion : str or torch.nn.modules, optional (default='mse')
		The loss function used to train the model.
		It can be custom loss class inherited from torch.nn.modules.
		Applicable when loss_func is None.

	device : str, optional (default=None)
		The device to use for the model. If None, it will be decided
		automatically. If you want to use MPS, set it to 'mps'.

	random_state : int, optional (default=42)
		The random seed for reproducibility.

	use_compile : bool, optional (default=False)
		Whether to compile the model.
		If True, the model will be compiled before training.
		This is only available for
		PyTorch version >= 2.0.0. and Python < 3.12.

	compile_mode : str, optional (default='default')
		The mode to compile the model.
		Can be either “default”, “reduce-overhead”,
		“max-autotune” or “max-autotune-no-cudagraphs”.
		See https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile for details.

	verbose : int, optional (default=1)
		Verbosity mode.
		- 0 = silent
		- 1 = progress bar
		- 2 = one line per epoch.

	optimizer_params : dict, optional (default=None)
		Additional parameters for the optimizer.
		For example, `optimizer_params={'weight_decay': 1e-4}`.

	criterion_params : dict, optional (default=None)
		Additional parameters for the criterion.
		For example, `criterion_params={'reduction': 'sum'}`.
	"""

	def __init__(self,
				 contamination=0.1, preprocessing=True,
				 lr=1e-3, epoch_num=10, batch_size=32,
				 optimizer='adam', loss_func=None, criterion='mse',
				 device=None, random_state=42,
				 use_compile=False, compile_mode='default',
				 verbose=1,
				 optimizer_params: dict = {},
				 criterion_params: dict = {}):
		super(BaseDeepLearningDetector, self).__init__(
			contamination=contamination)
		self.preprocessing = preprocessing
		self.lr = lr
		self.epoch_num = epoch_num
		self.batch_size = batch_size
		self.optimizer = optimizer
		self.loss_func = loss_func
		self.criterion = criterion
		self.device = device
		self.random_state = random_state
		self.use_compile = use_compile
		self.compile_mode = compile_mode
		self.verbose = verbose
		self.optimizer_params = optimizer_params
		self.criterion_params = criterion_params

		self.X_mean = None
		self.X_std = None

		if (isinstance(contamination, (float, int))):
			if not (0. < contamination <= 0.5):
				raise ValueError(f"contamination must be in (0., 0.5], "
								 f"got {contamination}")

		# set loss function or criterion
		if isfunction(loss_func):
			self.loss_func = loss_func
		elif loss_func is not None:
			raise ValueError('Invalid loss function.')
		if isinstance(criterion, str):
			self.criterion = get_criterion_by_name(name=criterion,
												   **self.criterion_params)
		elif isinstance(criterion, torch.nn.Module):
			self.criterion = criterion
		else:
			raise ValueError('Invalid criterion.')

		# set random seed for reproducibility
		self._set_seed(self.random_state)

		# decide device based on availablity
		if self.device is None:
			self.device = torch.device(
				"cuda:0" if torch.cuda.is_available() else "cpu")
			# If you want to use MPS, uncomment the following lines
			# self.device = torch.device(
			#     "mps" if torch.backends.mps.is_available() else self.device)

	def fit(self, X, y=None):
		"""Fit detector. y is ignored in unsupervised methods.

		Parameters
		----------
		X : numpy array of shape (n_samples, n_features)
			The input samples.

		y : numpy array of shape (n_samples,), optional (default=None)
			The ground truth of input samples. Not used in unsupervised methods.
		"""
		# validate inputs X and y (optional)
		X = check_array(X)
		if y is not None:
			self._set_n_classes(y)

		feature_size = X.shape[1]
		self.build_model(feature_size)
		self.training_prepare()

		if self.preprocessing:
			self.X_mean = np.mean(X, axis=0)
			self.X_std = np.std(X, axis=0)
			train_set = TorchDataset(X=X, y=y,
									 mean=self.X_mean, std=self.X_std)
		else:
			train_set = TorchDataset(X=X, y=y)

		# create data loader
		train_loader = torch.utils.data.DataLoader(
			dataset=train_set, batch_size=self.batch_size,
			shuffle=True, drop_last=True)

		# train the model
		self.train(train_loader)

		self.decision_scores_ = self.decision_function(X)
		self._process_decision_scores()

	def training_prepare(self):
		self.model = self.model.to(self.device)

		# set optimizer
		self.optimizer = get_optimizer_by_name(model=self.model,
											   name=self.optimizer,
											   lr=self.lr,
											   **self.optimizer_params)

		if self.use_compile:
			self.model = torch.compile(model=self.model,
									   mode=self.compile_mode)
			print('Model compiled.')

		self.model.train()

	def train(self, train_loader):
		"""Train the deep learning model.

		Parameters
		----------
		train_loader : torch.utils.data.DataLoader
			The data loader for training the model.
		"""
		for epoch in tqdm.trange(self.epoch_num,
								 desc=f'Training: ',
								 disable=not self.verbose == 1):
			start_time = time.time()
			overall_loss = []
			for batch_data in train_loader:
				loss = self.training_forward(batch_data)
				overall_loss.append(loss)
			overall_loss = np.mean(overall_loss)

			if self.verbose == 2:
				end_time = time.time()
				print(f'Epoch: {epoch + 1} / {self.epoch_num}, '
					  f'Loss: {overall_loss:.4f}, '
					  f'Time: {end_time - start_time:.2f}s')

			self.epoch_update()

	def decision_function(self, X, y=None):
		"""
		Predict raw anomaly score of X using the fitted detector.

		The anomaly score of an input sample is computed based on different
		detector algorithms. For consistency, outliers are assigned with
		larger anomaly scores.
		Parameters
		----------
		X : numpy array of shape (n_samples, n_features)
			The training input samples. Sparse matrices are accepted only
			if they are supported by the base estimator.
		Returns
		-------
		anomaly_scores : numpy array of shape (n_samples,)
			The anomaly score of the input samples.
		"""
		X = check_array(X)
		if self.preprocessing:
			dataset = TorchDataset(X=X, y=y, mean=self.X_mean, std=self.X_std)
		else:
			dataset = TorchDataset(X=X, y=y)

		data_loader = torch.utils.data.DataLoader(
			dataset=dataset, batch_size=self.batch_size, shuffle=False)

		# evaluate the model
		anomaly_scores = self.evaluate(data_loader)
		anomaly_scores = self.decision_function_update(anomaly_scores)
		return anomaly_scores

	def evaluate(self, data_loader):
		"""
		Evaluate the deep learning model.

		Parameters
		----------
		data_loader : torch.utils.data.DataLoader
			The data loader for evaluating the model.

		Returns
		-------
		outlier_scores : numpy array of shape (n_samples,)
			The outlier scores of the input samples.
		"""
		self.model.eval()
		anamoly_scores = []
		with torch.no_grad():
			for batch_data in data_loader:
				score = self.evaluating_forward(batch_data)
				anamoly_scores.append(score)
		anamoly_scores = np.concatenate(anamoly_scores)
		return anamoly_scores

	def save(self, path):
		"""Save the model to the specified path.

		Parameters
		----------
		path : str
			The path to save the model.
		"""
		# save the class
		with open(path, 'wb') as file:
			pickle.dump(self, file)

	@classmethod
	def load(cls, path):
		"""Load the model from the specified path.

		Parameters
		----------
		path : str
			The path to load the model.

		Returns
		-------
		model : BaseDeepLearningDetector
			The loaded model.
		"""
		with open(path, 'rb') as file:
			detector = pickle.load(file)
		return detector

	@staticmethod
	def _set_seed(random_state):
		"""Set random seed for reproducibility
		"""
		os.environ['PYTHONHASHSEED'] = str(random_state)
		random.seed(random_state)
		np.random.seed(random_state)
		torch.manual_seed(random_state)

	@abstractmethod
	def build_model(self, feature_size):
		"""
		Need to define `self.model` in this method.
		"""
		pass

	@abstractmethod
	def training_forward(self, batch_data):
		"""
		Forward pass for training the model.
		Abstract method to be implemented.

		Parameters
		----------
		batch_data : tuple
			The batch data for training the model.

		Returns
		-------
		loss : float
			The loss.item of the model.
		"""
		# An example implementation:
		# x = batch_data
		# x = x.to(self.device)
		# # x, y = batch_data
		# # x = x.to(self.device)
		# # y = y.to(self.device)
		# self.optimizer.zero_grad()
		# output = self.model(x)
		# if hasattr(self.model, 'loss_func'):
		#     loss = self.loss_func(output, x)
		# else:
		#     loss = self.criterion(output, x)
		# loss.backward()
		# self.optimizer.step()
		# return loss.item()
		pass

	@abstractmethod
	def evaluating_forward(self, batch_data):
		"""
		Forward pass for evaluating the model.
		Abstract method to be implemented.

		Parameters
		----------
		batch_data : tuple
			The batch data for evaluating the model.

		Returns
		-------
		output : numpy array
			The output of the model.
		"""
		# An example implementation:
		# x = batch_data
		# x_gpu = x.to(self.device)
		# # x, y = batch_data
		# # x_gpu = x.to(self.device)
		# # y = y.to(self.device)
		# output = self.model(x_gpu)
		# return pairwise_distances_no_broadcast(x.numpy(),
		#                                        output.cpu().numpy())
		pass

	def epoch_update(self):
		"""
		For any additional operations after each epoch.
		"""
		pass

	def decision_function_update(self, anomaly_scores):
		"""
		For any additional operations after each decision function call.
		"""
		return anomaly_scores
