# -*- coding: utf-8 -*-
"""IsolationForest Outlier Detector. Implemented on scikit-learn library.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from joblib import Parallel
from joblib.parallel import delayed
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.ensemble._iforest  import _average_path_length
from sklearn.utils.validation import  _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
import numpy as np 
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm

import pandas as pd 
import os 
import pickle
import time
import random

from .base import BaseDetector
# noinspection PyProtectedMember
from ..utils.utility import invert_order


# TODO: behavior of Isolation Forest will change in sklearn 0.22. See below.
# in 0.22, scikit learn will start adjust decision_function values by
# offset to make the values below zero as outliers. In other words, it is
# an absolute shift, which SHOULD NOT affect the result of PyOD at all as
# the order is still preserved.

# Behaviour of the decision_function which can be either ‘old’ or ‘new’.
# Passing behaviour='new' makes the decision_function change to match other
# anomaly detection algorithm API which will be the default behaviour in the
# future. As explained in details in the offset_ attribute documentation,
# the decision_function becomes dependent on the contamination parameter,
# in such a way that 0 becomes its natural threshold to detect outliers.

# offset_ : float
# Offset used to define the decision function from the raw scores.
# We have the relation: decision_function = score_samples - offset_.
# Assuming behaviour == ‘new’, offset_ is defined as follows.
# When the contamination parameter is set to “auto”,
# the offset is equal to -0.5 as the scores of inliers are close to 0 and the
# scores of outliers are close to -1. When a contamination parameter different
# than “auto” is provided, the offset is defined in such a way we obtain the
# expected number of outliers (samples with decision function < 0) in training.
# Assuming the behaviour parameter is set to ‘old’,
# we always have offset_ = -0.5, making the decision function independent from
# the contamination parameter.

# check https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html for more information


class IForest(BaseDetector):
	"""Wrapper of scikit-learn Isolation Forest with more functionalities.

	The IsolationForest 'isolates' observations by randomly selecting a
	feature and then randomly selecting a split value between the maximum and
	minimum values of the selected feature.
	See :cite:`liu2008isolation,liu2012isolation` for details.

	Since recursive partitioning can be represented by a tree structure, the
	number of splittings required to isolate a sample is equivalent to the path
	length from the root node to the terminating node.

	This path length, averaged over a forest of such random trees, is a
	measure of normality and our decision function.

	Random partitioning produces noticeably shorter paths for anomalies.
	Hence, when a forest of random trees collectively produce shorter path
	lengths for particular samples, they are highly likely to be anomalies.

	Parameters
	----------
	n_estimators : int, optional (default=100)
		The number of base estimators in the ensemble.

	max_samples : int or float, optional (default="auto")
		The number of samples to draw from X to train each base estimator.

			- If int, then draw `max_samples` samples.
			- If float, then draw `max_samples * X.shape[0]` samples.
			- If "auto", then `max_samples=min(256, n_samples)`.

		If max_samples is larger than the number of samples provided,
		all samples will be used for all trees (no sampling).

	contamination : float in (0., 0.5), optional (default=0.1)
		The amount of contamination of the data set, i.e. the proportion
		of outliers in the data set. Used when fitting to define the threshold
		on the decision function.

	max_features : int or float, optional (default=1.0)
		The number of features to draw from X to train each base estimator.

			- If int, then draw `max_features` features.
			- If float, then draw `max_features * X.shape[1]` features.

	bootstrap : bool, optional (default=False)
		If True, individual trees are fit on random subsets of the training
		data sampled with replacement. If False, sampling without replacement
		is performed.

	n_jobs : integer, optional (default=1)
		The number of jobs to run in parallel for both `fit` and `predict`.
		If -1, then the number of jobs is set to the number of cores.

	behaviour : str, default='old'
		Behaviour of the ``decision_function`` which can be either 'old' or
		'new'. Passing ``behaviour='new'`` makes the ``decision_function``
		change to match other anomaly detection algorithm API which will be
		the default behaviour in the future. As explained in details in the
		``offset_`` attribute documentation, the ``decision_function`` becomes
		dependent on the contamination parameter, in such a way that 0 becomes
		its natural threshold to detect outliers.

		.. versionadded:: 0.7.0
		   ``behaviour`` is added in 0.7.0 for back-compatibility purpose.

		.. deprecated:: 0.20
		   ``behaviour='old'`` is deprecated in sklearn 0.20 and will not be
		   possible in 0.22.

		.. deprecated:: 0.22
		   ``behaviour`` parameter will be deprecated in sklearn 0.22 and
		   removed in 0.24.

		.. warning::
			Only applicable for sklearn 0.20 above.

	random_state : int, RandomState instance or None, optional (default=None)
		If int, random_state is the seed used by the random number generator;
		If RandomState instance, random_state is the random number generator;
		If None, the random number generator is the RandomState instance used
		by `np.random`.

	verbose : int, optional (default=0)
		Controls the verbosity of the tree building process.

	Attributes
	----------
	estimators_ : list of DecisionTreeClassifier
		The collection of fitted sub-estimators.

	estimators_samples_ : list of arrays
		The subset of drawn samples (i.e., the in-bag samples) for each base
		estimator.

	max_samples_ : integer
		The actual number of samples

	decision_scores_ : numpy array of shape (n_samples,)
		The outlier scores of the training data.
		The higher, the more abnormal. Outliers tend to have higher
		scores. This value is available once the detector is
		fitted.

	threshold_ : float
		The threshold is based on ``contamination``. It is the
		``n_samples * contamination`` most abnormal samples in
		``decision_scores_``. The threshold is calculated for generating
		binary outlier labels.

	labels_ : int, either 0 or 1
		The binary labels of the training data. 0 stands for inliers
		and 1 for outliers/anomalies. It is generated by applying
		``threshold_`` on ``decision_scores_``.
	"""

	def __init__(self, n_estimators=100,
				 max_samples="auto",
				 contamination=0.1,
				 max_features=1.,
				 bootstrap=False,
				 n_jobs=1,
				 behaviour='old',
				 random_state=None,
				 verbose=0):
		super(IForest, self).__init__(contamination=contamination)
		self.n_estimators = n_estimators
		self.max_samples = max_samples
		self.max_features = max_features
		self.bootstrap = bootstrap
		self.n_jobs = n_jobs
		self.behaviour = behaviour
		self.random_state = random_state
		self.verbose = verbose

	def fit(self, X, y=None):
		"""Fit detector. y is ignored in unsupervised methods.

		Parameters
		----------
		X : numpy array of shape (n_samples, n_features)
			The input samples.

		y : Ignored
			Not used, present for API consistency by convention.

		Returns
		-------
		self : object
			Fitted estimator.
		"""
		# validate inputs X and y (optional)
		X = check_array(X)
		self._set_n_classes(y)

		# In sklearn 0.20+ new behaviour is added (arg behaviour={'new','old'})
		# to IsolationForest that shifts the location of the anomaly scores
		# noinspection PyProtectedMember

		self.detector_ = IsolationForest(n_estimators=self.n_estimators,
										 max_samples=self.max_samples,
										 contamination=self.contamination,
										 max_features=self.max_features,
										 bootstrap=self.bootstrap,
										 n_jobs=self.n_jobs,
										 random_state=self.random_state,
										 verbose=self.verbose)

		self.detector_.fit(X=X, y=None, sample_weight=None)

		# invert decision_scores_. Outliers comes with higher outlier scores.
		self.decision_scores_ = invert_order(
			self.detector_.decision_function(X))
		self._process_decision_scores()
		return self

	def decision_function(self, X):
		"""Predict raw anomaly score of X using the fitted detector.

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
		check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
		# invert outlier scores. Outliers comes with higher outlier scores
		return invert_order(self.detector_.decision_function(X))

	@property
	def estimators_(self):
		"""The collection of fitted sub-estimators.
		Decorator for scikit-learn Isolation Forest attributes.
		"""
		return self.detector_.estimators_

	@property
	def estimators_samples_(self):
		"""The subset of drawn samples (i.e., the in-bag samples) for
		each base estimator.
		Decorator for scikit-learn Isolation Forest attributes.
		"""
		return self.detector_.estimators_samples_

	@property
	def max_samples_(self):
		"""The actual number of samples.
		Decorator for scikit-learn Isolation Forest attributes.
		"""
		return self.detector_.max_samples_
	
	@property
	def _max_features(self):
		"""The number of features used by the model (i.e. self.max_features * X.shape[1]).
		Decorator for scikit-learn Isolation Forest attributes.
		"""
		return self.detector_._max_features

	@property
	def estimators_features_(self):
		"""The indeces of the subset of features used to train the estimators.
		Decorator for scikit-learn Isolation Forest attributes.
		"""
		return self.detector_.estimators_features_

	@property
	def n_features_in_(self):
		"""The number of features seen during the fit.
		Decorator for scikit-learn Isolation Forest attributes.
		"""
		return self.detector_.n_features_in_

	@property
	def offset_(self):
		"""Offset used to define the decision function from the raw scores.
		Decorator for scikit-learn Isolation Forest attributes.
		"""
		return self.detector_.offset_

	@property
	def feature_importances_(self):
		"""The impurity-based feature importance. The higher, the more
		important the feature. The importance of a feature is computed as the
		(normalized) total reduction of the criterion brought by that feature.
		It is also known as the Gini importance.

		.. warning::
		impurity-based feature importance can be misleading for
		high cardinality features (many unique values). See
		https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
		as an alternative.

		Returns
		-------
		feature_importances_ : ndarray of shape (n_features,)
			The values of this array sum to 1, unless all trees are single node
			trees consisting of only the root node, in which case it will be an
			array of zeros.
		"""
		check_is_fitted(self)
		all_importances = Parallel(
			n_jobs=self.n_jobs)(
			delayed(getattr)(tree, "feature_importances_")
			for tree in self.detector_.estimators_
			if tree.tree_.node_count > 1
		)

		if not all_importances:
			return np.zeros(self.n_features_in_, dtype=np.float64)

		all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
		return all_importances / np.sum(all_importances)
	
	# The functions below have been adapted from the sklearn source code

	def decision_function_single_tree(self, tree_idx, X):
		"""Modification of the decision_function method from sklearn.ensemble.IsolationForest which compute the decision function 
		for a single tree of the forest. 

		Parameters
		----------
		tree_idx : Index of the iTree on which the decision function is computed. 
		X : numpy array of shape (n_samples, n_features) representing the in-bag sample of the tree. 

		Returns
		-------
		decision_function : numpy array of shape (n_samples,) representing the decision function of the tree on its in-bag sample.
		"""
		return self._score_samples(tree_idx, X) - self.offset_


	def _score_samples(self, tree_idx, X):
		"""Modification of the score_samples method from sklearn.ensemble.IsolationForest which compute the score samples for a single tree of the forest. 

		Parameters
		----------
		tree_idx : Index of the iTree on which the decision function is computed. 
		X : numpy array of shape (n_samples, n_features) representing the in-bag sample of the tree. 

		Returns
		-------
		score_samples : numpy array of shape (n_samples,) representing the score samples of the tree on its in-bag sample.
		"""
		n_feat= self.n_features_in_
		if n_feat != X.shape[1]:
			raise ValueError("Number of features of the model must "
							"match the input. Model n_features is {0} and "
							"input n_features is {1}."
							"".format(n_feat, X.shape[1]))
		return -self._compute_chunked_score_samples(tree_idx, X)


	def _compute_chunked_score_samples(self, tree_idx, X):
		"""Modification of the compute_chunked_score_samples method from sklearn.ensemble.IsolationForest 
		used to compute the score samples on the maximum number of rows processable by the working memory for a single tree of the forest. 

		Parameters
		----------
		tree_idx : Index of the iTree on which the decision function is computed. 
		X : numpy array of shape (n_samples, n_features) representing the in-bag sample of the tree. 

		Returns
		-------
		score_samples : numpy array of shape (n_samples,) representing the score samples of the tree on a batch of the in-bag sample.
		"""
		n_samples = _num_samples(X)
		if int(self.max_features*X.shape[1]) == X.shape[1]:
			subsample_features = False
		else:
			subsample_features = True
		chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self._max_features,
										max_n_rows=n_samples)
		slices = gen_batches(n_samples, chunk_n_rows)
		scores = np.zeros(n_samples, order="f")
		for sl in slices:
			scores[sl] = self._compute_score_samples_single_tree(tree_idx, X[sl], subsample_features)
		return scores


	def _compute_score_samples_single_tree(self, tree_idx, X, subsample_features):
		"""Modification of the _compute_score_samples method from sklearn.ensemble.IsolationForest 
		used to compute the score samples for each sample for a single tree of the forest. 

		Parameters
		----------
		tree_idx : Index of the iTree on which the decision function is computed. 
		X : numpy array of shape (n_samples, n_features) representing the in-bag sample of the tree.
		subsample_features : boolean indicating if the tree has been trained on a subsample of the features.

		Returns
		-------
		score_samples : numpy array of shape (n_samples,) representing the score samples of the tree in its in-bag sample.
		"""
		n_samples = X.shape[0]
		depths = np.zeros(n_samples, order="f")
		tree = self.estimators_[tree_idx]
		features = self.estimators_features_[tree_idx]
		X_subset = X[:, features] if subsample_features else X
		leaves_index = tree.apply(X_subset)
		node_indicator = tree.decision_path(X_subset)
		n_samples_leaf = tree.tree_.n_node_samples[leaves_index]
		depths += (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
		scores = 2 ** (-depths / (1 * _average_path_length([self.max_samples_])))
		return scores
	
	def fs_datasets_hyperparams(self,dataset):
		"""Returns a list of hyperparametr values to train the iForest model for different datasets. 

		Parameters
		----------
		dataset : Dataset name. Available names are: 'cardio', 'ionosphere', 'lympho', 'letter', 'musk', 'satellite'.
		"""
		data = {
				# cardio
				('cardio'): {'contamination': 0.1, 'max_samples': 64, 'n_estimators': 150},
				# ionosphere
				('ionosphere'): {'contamination': 0.2, 'max_samples': 256, 'n_estimators': 100},
				# lympho
				('lympho'): {'contamination': 0.05, 'max_samples': 64, 'n_estimators': 150},
				# letter
				('letter'):  {'contamination': 0.1, 'max_samples': 256, 'n_estimators': 50},
				# musk
				('musk'): {'contamination': 0.05, 'max_samples': 128, 'n_estimators': 100},
				# satellite
				('satellite'): {'contamination': 0.15, 'max_samples': 64, 'n_estimators': 150}
				}
		return data[dataset]
	
	def diffi_ib(self, X, adjust_iic=True): # "ib" stands for "in-bag"
		"""Computes the Global Feature Importance scores for a set of input samples according to the DIFFI algorithm. 

		Parameters
		----------
		X : numpy array of shape (n_samples, n_features) representing the input samples.
		adjust_iic : boolean indicating if the IICs (Induced Imbalance Coefficients) should be adjusted or not.

		Returns
		-------
		fi_ib : numpy array of shape (n_features,) representing the Global Feature Importance scores.
		exec_time : float representing the execution time of the algorithm.
		"""
		# start time
		start = time.time()
		# initialization
		num_feat = X.shape[1] 
		estimators = self.estimators_
		cfi_outliers_ib = np.zeros(num_feat).astype('float')
		cfi_inliers_ib = np.zeros(num_feat).astype('float')
		counter_outliers_ib = np.zeros(num_feat).astype('int')
		counter_inliers_ib = np.zeros(num_feat).astype('int')
		in_bag_samples = self.estimators_samples_
		# for every iTree in the iForest
		for k, estimator in enumerate(estimators):
			# get in-bag samples indices
			in_bag_sample = list(in_bag_samples[k])
			# get in-bag samples (predicted inliers and predicted outliers)
			X_ib = X[in_bag_sample,:]
			as_ib = self.decision_function_single_tree(k, X_ib)
			X_outliers_ib = X_ib[np.where(as_ib < 0)]
			X_inliers_ib = X_ib[np.where(as_ib > 0)]
			if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0:
				continue
			# compute relevant quantities
			n_nodes = estimator.tree_.node_count
			children_left = estimator.tree_.children_left
			children_right = estimator.tree_.children_right
			feature = estimator.tree_.feature
			node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
			is_leaves = np.zeros(shape=n_nodes, dtype=bool)
			# compute node depths
			stack = [(0, -1)]  
			while len(stack) > 0:
				node_id, parent_depth = stack.pop()
				node_depth[node_id] = parent_depth + 1
				# if we have a test node
				if (children_left[node_id] != children_right[node_id]):
					stack.append((children_left[node_id], parent_depth + 1))
					stack.append((children_right[node_id], parent_depth + 1))
				else:
					is_leaves[node_id] = True
			# OUTLIERS
			# compute IICs for outliers
			lambda_outliers_ib = self._get_iic(estimator, X_outliers_ib, is_leaves, adjust_iic)
			# update cfi and counter for outliers
			node_indicator_all_points_outliers_ib = estimator.decision_path(X_outliers_ib)
			node_indicator_all_points_array_outliers_ib = node_indicator_all_points_outliers_ib.toarray()
			# for every point judged as abnormal
			for i in range(len(X_outliers_ib)):
				path = list(np.where(node_indicator_all_points_array_outliers_ib[i] == 1)[0])
				depth = node_depth[path[-1]]
				for node in path:
					current_feature = feature[node]
					if lambda_outliers_ib[node] == -1:
						continue
					else:
						cfi_outliers_ib[current_feature] += (1 / depth) * lambda_outliers_ib[node]
						counter_outliers_ib[current_feature] += 1
			# INLIERS
			# compute IICs for inliers 
			lambda_inliers_ib = self._get_iic(estimator, X_inliers_ib, is_leaves, adjust_iic)
			# update cfi and counter for inliers
			node_indicator_all_points_inliers_ib = estimator.decision_path(X_inliers_ib)
			node_indicator_all_points_array_inliers_ib = node_indicator_all_points_inliers_ib.toarray()
			# for every point judged as normal
			for i in range(len(X_inliers_ib)):
				path = list(np.where(node_indicator_all_points_array_inliers_ib[i] == 1)[0])
				depth = node_depth[path[-1]]
				for node in path:
					current_feature = feature[node]
					if lambda_inliers_ib[node] == -1:
						continue
					else:
						cfi_inliers_ib[current_feature] += (1 / depth) * lambda_inliers_ib[node]
						counter_inliers_ib[current_feature] += 1
		# compute FI
		fi_outliers_ib = np.where(counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0)
		fi_inliers_ib = np.where(counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0)
		fi_ib = fi_outliers_ib / fi_inliers_ib
		end = time.time()
		exec_time = end - start
		return fi_ib, exec_time


	def local_diffi(self, x):
		"""Compute the Local Feature Importance scores for a single input sample according to the DIFFI algorithm.

		Parameters
		----------
		x : numpy array of shape (n_features,) representing the input sample.

		Returns
		-------
		fi : numpy array of shape (n_features,) representing the Local Feature Importance scores.
		exec_time : float representing the execution time of the algorithm.
		"""
		# start time
		start = time.time()
		# initialization 
		estimators = self.estimators_
		cfi = np.zeros(len(x)).astype('float')
		counter = np.zeros(len(x)).astype('int')
		max_depth = int(np.ceil(np.log2(self.max_samples)))
		# for every iTree in the iForest
		for estimator in estimators:
			n_nodes = estimator.tree_.node_count
			children_left = estimator.tree_.children_left
			children_right = estimator.tree_.children_right
			feature = estimator.tree_.feature
			node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
			is_leaves = np.zeros(shape=n_nodes, dtype=bool)
			# compute node depths
			stack = [(0, -1)]  
			while len(stack) > 0:
				node_id, parent_depth = stack.pop()
				node_depth[node_id] = parent_depth + 1
				# if test node
				if (children_left[node_id] != children_right[node_id]):
					stack.append((children_left[node_id], parent_depth + 1))
					stack.append((children_right[node_id], parent_depth + 1))
				else:
					is_leaves[node_id] = True
			# update cumulative importance and counter
			x = x.reshape(1,-1)
			node_indicator = estimator.decision_path(x)
			node_indicator_array = node_indicator.toarray()
			path = list(np.where(node_indicator_array == 1)[1])
			leaf_depth = node_depth[path[-1]]
			for node in path:
				if not is_leaves[node]:
					current_feature = feature[node] 
					cfi[current_feature] += (1 / leaf_depth) - (1 / max_depth)
					counter[current_feature] += 1
		# compute FI
		fi = np.zeros(len(cfi))
		for i in range(len(cfi)):
			if counter[i] != 0:
				fi[i] = cfi[i] / counter[i]
		end = time.time()
		exec_time = end - start
		return fi, exec_time


	def _get_iic(self,estimator, predictions, is_leaves, adjust_iic):
		"""Computes the Induced Imbalance Coefficients (IIC) for a tree of the iForest.

		Parameters
		----------
		estimator : Tree of the iForest.
		predictions : Subset of the initial training set, containing the inliers or the outliers, on which the IIC are computed.
		is_leaves : Boolean array of shape (n_nodes,) indicating if a node is a leaf or not.
		adjust_iic : Boolean indicating if the IIC should be adjusted or not.

		Returns
		-------
		lambda_ : numpy array of shape (n_nodes,) representing the IIC for each node of the tree.
		"""
		desired_min = 0.5
		desired_max = 1.0
		epsilon = 0.0
		n_nodes = estimator.tree_.node_count
		lambda_ = np.zeros(n_nodes)
		children_left = estimator.tree_.children_left
		children_right = estimator.tree_.children_right
		# compute samples in each node
		node_indicator_all_samples = estimator.decision_path(predictions).toarray() 
		num_samples_in_node = np.sum(node_indicator_all_samples, axis=0)
		# ASSIGN INDUCED IMBALANCE COEFFICIENTS (IIC)
		for node in range(n_nodes):
			# compute relevant quantities for current node
			num_samples_in_current_node = num_samples_in_node[node]
			num_samples_in_left_children = num_samples_in_node[children_left[node]]
			num_samples_in_right_children = num_samples_in_node[children_right[node]]        
			# if there is only 1 feasible split or node is leaf -> no IIC is assigned
			if num_samples_in_current_node == 0 or num_samples_in_current_node == 1 or is_leaves[node]:    
				lambda_[node] = -1         
			# if useless split -> assign epsilon
			elif num_samples_in_left_children == 0 or num_samples_in_right_children == 0:
				lambda_[node] = epsilon
			else:
				if num_samples_in_current_node%2==0:    # even
					current_min = 0.5
				else:   # odd
					current_min = ceil(num_samples_in_current_node/2)/num_samples_in_current_node
				current_max = (num_samples_in_current_node-1)/num_samples_in_current_node
				tmp = np.max([num_samples_in_left_children, num_samples_in_right_children]) / num_samples_in_current_node
				if adjust_iic and current_min!=current_max:
					lambda_[node] = ((tmp-current_min)/(current_max-current_min))*(desired_max-desired_min)+desired_min
				else:
					lambda_[node] = tmp
		return lambda_
	
	def local_diffi_batch(self, X):
		"""Computes the Local Feature Importance scores for a set of input samples according to the DIFFI algorithm.

		Parameters
		----------
		X : numpy array of shape (n_samples, n_features) representing the input samples.

		Returns
		-------
		fi : numpy array of shape (n_samples, n_features) representing the Local Feature Importance scores.
		ord_idx : numpy array of shape (n_samples, n_features) representing the order of the features according to their Local Feature Importance scores.
		The samples are sorted in decreasing order of Feature Importance. 
		exec_time : float representing the execution time of the algorithm.
		"""
		fi = []
		ord_idx = []
		exec_time = []
		for i in range(X.shape[0]):
			x_curr = X[i, :]
			fi_curr, exec_time_curr = self.local_diffi(x_curr)
			fi.append(fi_curr)
			ord_idx_curr = np.argsort(fi_curr)[::-1]
			ord_idx.append(ord_idx_curr)
			exec_time.append(exec_time_curr)
		fi = np.vstack(fi)
		ord_idx = np.vstack(ord_idx)
		return fi, ord_idx, exec_time
	


	
	def compute_local_importances(self,X: pd.DataFrame,name: str,pwd_imp_score: str = os.getcwd(), pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
		"""
		Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
		functions. 
		
		Parameters
		----------
		X: Input dataset   
		name: Dataset's name   
		pwd_imp_score: Directory where the Importance Scores results will be saved as pkl files, by default the current working directory
		pwd_plt_data: Directory where the plot data results will be saved as pkl files, by default the current working directory        
	
		Returns
		----------
		imps: array of shape (n_samples,n_features) containing the local Feature Importance values for the samples of the input dataset X. 
		The array is also locally saved in a pkl file for the sake of reproducibility.
		plt_data: Dictionary containig the average Importance Scores values, the feature order and the standard deviations on the Importance Scores. 
		The dictionary is also locally saved in a pkl file for the sake of reproducibility.
		path_fi: Path of the pkl file containing the Importance Scores.
		path_plt_data: Path of the pkl file containing the plt data.    
		"""

		name='LFI_'+name
		fi,_,_=self.local_diffi_batch(X)

		# Handle the case in which there are some np.nan or np.inf values in the fi array
		if np.isnan(fi).any() or np.isinf(fi).any():
			#Substitute the np.nan values with 0 and the np.inf values with the maximum value of the fi array plus 1. 
			fi=np.nan_to_num(fi,nan=0,posinf=np.nanmax(fi[np.isfinite(fi)])+1)
		
		# Save the Importance Scores in a pkl file
		path_fi = pwd_imp_score  + '/imp_scores_' + name + '.pkl'
		with open(path_fi, 'wb') as fl:
			pickle.dump(fi,fl)

		""" 
		Take the mean feature importance scores over the different runs for the Feature Importance Plot
		and put it in decreasing order of importance.
		To remove the possible np.nan or np.inf values from the mean computation use assign np.nan to the np.inf values 
		and then ignore the np.nan values using np.nanmean
		"""

		fi[fi==np.inf]=np.nan
		mean_imp=np.nanmean(fi,axis=0)
		std_imp=np.nanstd(fi,axis=0)
		mean_imp_val=np.sort(mean_imp)
		feat_order=mean_imp.argsort()

		plt_data={'Importances': mean_imp_val,
				'feat_order': feat_order,
				'std': std_imp[mean_imp.argsort()]}
		
		# Save the plt_data dictionary in a pkl file
		path_plt_data = pwd_plt_data + '/plt_data_' + name + '.pkl'
		with open(path_plt_data, 'wb') as fl:
			pickle.dump(plt_data,fl)
		

		return fi,plt_data,path_fi,path_plt_data
	
	def compute_global_importances(self,X: pd.DataFrame, n_runs:int, name: str,pwd_imp_score: str = os.getcwd(), pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
		"""
		Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
		functions. 
		
		Parameters
		----------
		X: Input Dataset
		n_runs: Number of runs to perform in order to compute the Global Feature Importance Scores.
		name: Dataset's name   
		pwd_imp_score: Directory where the Importance Scores results will be saved as pkl files, by default the current working directory
		pwd_plt_data: Directory where the plot data results will be saved as pkl files, by default the current working directory        
					
		Returns
		----------
		imps: array of shape (n_samples,n_features) containing the local Feature Importance values for the samples of the input dataset X. 
		The array is also locally saved in a pkl file for the sake of reproducibility.
		plt_data: Dictionary containing the average Importance Scores values, the feature order and the standard deviations on the Importance Scores. 
		The dictionary is also locally saved in a pkl file for the sake of reproducibility.
		path_fi: Path of the pkl file containing the Importance Scores
		path_plt_data: Path of the pkl file containing the plt data    
		"""

		name='GFI_'+name
		fi=np.zeros(shape=(n_runs,X.shape[1]))
		for i in range(n_runs):
			self.fit(X)
			fi[i,:],_=self.diffi_ib(X)

		# Handle the case in which there are some np.nan or np.inf values in the fi array
		if np.isnan(fi).any() or np.isinf(fi).any():
			#Substitute the np.nan values with 0 and the np.inf values with the maximum value of the fi array plus 1. 
			fi=np.nan_to_num(fi,nan=0,posinf=np.nanmax(fi[np.isfinite(fi)])+1)

		# Save the Importance Scores in a pkl file
		path_fi = pwd_imp_score + '/imp_scores_'  + name + '.pkl'
		with open(path_fi, 'wb') as fl:
			pickle.dump(fi,fl)
			

		fi[fi==np.inf]=np.nan
		mean_imp=np.nanmean(fi,axis=0)
		std_imp=np.nanstd(fi,axis=0)
		mean_imp_val=np.sort(mean_imp)
		feat_order=mean_imp.argsort()

		plt_data={'Importances': mean_imp_val,
				'feat_order': feat_order,
				'std': std_imp[mean_imp.argsort()]}
		
		# Save the plt_data dictionary in a pkl file
		path_plt_data = pwd_plt_data  +  '/plt_data_' + name + '.pkl'
		with open(path_plt_data, 'wb') as fl:
			pickle.dump(plt_data,fl)
		

		return fi,plt_data,path_fi,path_plt_data
	
	def plt_importances_bars(self,imps_path: str, name: str, pwd: str =os.getcwd(),f: int = 6,is_local: bool=False, save: bool =True):
		"""
		Obtain the Global Importance Bar Plot given the Importance Scores values computed in the compute_local_importance or compute_global_importance functions. 
		
		Parameters
		----------
		imps_path: Path of the pkl file containing the array of shape (n_samples,n_features) with the LFI/GFI Scores for the input dataset.
		Obtained from the compute_local_importance or compute_global_importance functions.   
		name: Dataset's name 
		pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.    
		f: Number of vertical bars to include in the Bar Plot. By default f is set to 6. 
		is_local: Boolean variable used to specify weather we are plotting the Global or Local Feature Importance in order to set the file name.
		If is_local is True the result will be the LFI Score Plot (based on the LFI scores of the input samples), otherwise the result is the GFI 
		Score Plot (based on the GFI scores obtained in the different n_runs execution of the model). By default is_local is set to False.  
		save: Boolean variable used to decide weather to save the Bar Plot locally as a PDF or not. BY default save is set to True. 
		
		Returns
		----------
		fig,ax : plt.figure  and plt.axes objects used to create the plot 
		bars: pd.DataFrame containing the percentage count of the features in the first f positions of the Bar Plot.    
		"""

		name_file='GFI_Bar_plot_'+name 

		if is_local:
			name_file='LFI_Bar_plot_'+name
		
		#Load the imps array from the pkl file contained in imps_path -> the imps_path is returned from the 
		#compute_local_importances or compute_global_importances functions so we have it for free 
		with open(imps_path, 'rb') as file:
			importances = pickle.load(file)

		number_colours = 20
		color = plt.cm.get_cmap('tab20',number_colours).colors
		patterns=[None,'!','@','#','$','^','&','*','°','(',')','-','_','+','=','[',']','{','}',
          '|',';',':','\l',',','.','<','>','/','?','`','~','\\','!!','@@','##','$$','^^','&&','**','°°','((']
		importances_matrix = np.array([np.array(pd.Series(x).sort_values(ascending = False).index).T for x in importances])
		dim=importances.shape[1]
		dim=int(dim)
		bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
		bars = pd.DataFrame(bars)

		tick_names=[]
		for i in range(1,f+1):
			if i==1:
				tick_names.append(r'${}'.format(i) + r'^{st}$')
			elif i==2:
				tick_names.append(r'${}'.format(i) + r'^{nd}$')
			elif i==3:
				tick_names.append(r'${}'.format(i) + r'^{rd}$')
			else:
				tick_names.append(r'${}'.format(i) + r'^{th}$')

		barWidth = 0.85
		r = range(dim)
		ncols=1
		if importances.shape[1]>15:
			ncols=2
		elif importances.shape[1]>30:
			ncols=3
		elif importances.shape[1]>45:
			ncols=4
		elif importances.shape[1]>60:
			ncols=5
		elif importances.shape[1]>75:
			ncols=6

		fig, ax = plt.subplots()

		for i in range(dim):
			ax.bar(r[:f], bars.T.iloc[i, :f].values, bottom=bars.T.iloc[:i, :f].sum().values, color=color[i % number_colours], edgecolor='white', width=barWidth, label=str(i), hatch=patterns[i // number_colours])

		ax.set_xlabel("Rank", fontsize=20)
		ax.set_xticks(range(f), tick_names[:f])
		ax.set_ylabel("Percentage count", fontsize=20)
		ax.set_yticks(range(10, 101, 10), [str(x) + "%" for x in range(10, 101, 10)])
		ax.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left",ncol=ncols)

		if save:
			plt.savefig(pwd + '/{}_bar_plot.pdf'.format(name_file), bbox_inches='tight')

		return fig, ax, bars


	def plt_feat_bar_plot(self,plt_data_path: str,name: str,pwd: str =os.getcwd(),is_local: bool =False,save: bool =True):
		"""
		Obtain the Global Feature Importance Score Plot exploiting the information obtained from the compute_local_importance or compute_global_importance functions. 
		
		Parameters
		----------
		plt_data_path: Dictionary generated from the compute_local_importance or compute_global_importance functions 
		with the necessary information to create the Score Plot.
		name: Dataset's name
		pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.  
		is_local: Boolean variable used to specify weather we are plotting the Global or Local Feature Importance in order to set the file name.
		If is_local is True the result will be the LFI Score Plot (based on the LFI scores of the input samples), otherwise the result is the GFI 
		Score Plot (based on the GFI scores obtained in the different n_runs execution of the model). By default is_local is set to False. 
		save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
					
		Returns
		----------
		ax1,ax2: The two plt.axes objects used to create the plot.  
		"""
		#Load the plt_data dictionary from the pkl file contained in plt_data_path -> the plt_data_path is returned from the 
		#compute_local_importances or compute_global_importances functions so we have it for free 
		with open(plt_data_path, 'rb') as f:
			plt_data = pickle.load(f)

		name_file='GFI_Score_plot_'+name 

		if is_local:
			name_file='LFI_Score_plot_'+name

		patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
		imp_vals=plt_data['Importances']
		feat_imp=pd.DataFrame({'Global Importance': np.round(imp_vals,3),
							'Feature': plt_data['feat_order'],
							'std': plt_data['std']
							})
		
		if len(feat_imp)>15:
			feat_imp=feat_imp.iloc[-15:].reset_index(drop=True)
		
		dim=feat_imp.shape[0]

		number_colours = 20

		plt.style.use('default')
		plt.rcParams['axes.facecolor'] = '#F2F2F2'
		plt.rcParams['axes.axisbelow'] = True
		color = plt.cm.get_cmap('tab20',number_colours).colors
		ax1=feat_imp.plot(y='Global Importance',x='Feature',kind="barh",color=color[feat_imp['Feature']%number_colours],xerr='std',
						capsize=5, alpha=1,legend=False,
						hatch=[patterns[i//number_colours] for i in feat_imp['Feature']])
		xlim=np.min(imp_vals)-0.2*np.min(imp_vals)

		ax1.grid(alpha=0.7)
		ax2 = ax1.twinx()
		# Add labels on the right side of the bars
		values=[]
		for i, v in enumerate(feat_imp['Global Importance']):
			values.append(str(v) + ' +- ' + str(np.round(feat_imp['std'][i],2)))
		
		ax2.set_ylim(ax1.get_ylim())
		ax2.set_yticks(range(dim))
		ax2.set_yticklabels(values)
		ax2.grid(alpha=0)
		plt.axvline(x=0, color=".5")
		ax1.set_xlabel('Importance Score',fontsize=20)
		ax1.set_ylabel('Features',fontsize=20)
		plt.xlim(xlim)
		plt.subplots_adjust(left=0.3)
		if save:
			plt.savefig(pwd+'/{}.pdf'.format(name_file),bbox_inches='tight')
			
		return ax1,ax2


	def plot_importance_map(self,name: str, X_train: pd.DataFrame,y_train: np.array ,resolution: int,
							pwd: str =os.getcwd(),save: bool =True,m: bool =None,factor: int =3,feats_plot: tuple[int,int] =(0,1),ax=None,labels: bool=True):
		"""
		Produce the Local Feature Importance Scoremap.   
		
		Parameters
		----------
		name: Dataset's name
		X_train: Training Set 
		y_train: Dataset training labels
		resolution: Scoremap resolution 
		pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
		save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
		m: Boolean variable regulating the plt.pcolor advanced settings. By defualt the value of m is set to None.
		factor: Integer factor used to define the minimum and maximum value of the points used to create the scoremap. By default the value of f is set to 3.
		feats_plot: This tuple contains the indexes of the pair features to compare in the Scoremap. By default the value of feats_plot
				is set to (0,1).
		ax: plt.axes object used to create the plot. By default ax is set to None.
		labels: Boolean variable used to decide weather to include the x and y label name in the plot.
		When calling the plot_importance_map function inside plot_complete_scoremap this parameter will be set to False 
					
		Returns
		----------
		fig,ax : plt.figure  and plt.axes objects used to create the plot 
		"""
		mins = X_train.min(axis=0)[list(feats_plot)]
		maxs = X_train.max(axis=0)[list(feats_plot)]  
		mean = X_train.mean(axis = 0)
		mins = list(mins-(maxs-mins)*factor/10)
		maxs = list(maxs+(maxs-mins)*factor/10)
		xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
		mean = np.repeat(np.expand_dims(mean,0),len(xx)**2,axis = 0)
		mean[:,feats_plot[0]]=xx.reshape(len(xx)**2)
		mean[:,feats_plot[1]]=yy.reshape(len(yy)**2)

		importance_matrix = np.zeros_like(mean)
		self.max_samples = len(X_train)
		for i in range(importance_matrix.shape[0]):
			importance_matrix[i] = self.local_diffi(mean[i])[0]
		
		sign = np.sign(importance_matrix[:,feats_plot[0]]-importance_matrix[:,feats_plot[1]])
		Score = sign*((sign>0)*importance_matrix[:,feats_plot[0]]+(sign<0)*importance_matrix[:,feats_plot[1]])
		x = X_train[:,feats_plot[0]].squeeze()
		y = X_train[:,feats_plot[1]].squeeze()
		
		Score = Score.reshape(xx.shape)

		# Create a new pyplot object if plt is not provided
		if ax is None:
			fig, ax = plt.subplots()
		
		if m is not None:
			cp = ax.pcolor(xx, yy, Score, cmap=cm.RdBu, vmin=-m, vmax=m, shading='nearest')
		else:
			cp = ax.pcolor(xx, yy, Score, cmap=cm.RdBu, shading='nearest', norm=colors.CenteredNorm())
		
		ax.contour(xx, yy, (importance_matrix[:, feats_plot[0]] + importance_matrix[:, feats_plot[1]]).reshape(xx.shape), levels=7, cmap=cm.Greys, alpha=0.7)

		try:
			ax.scatter(x[y_train == 0], y[y_train == 0], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
			ax.scatter(x[y_train == 1], y[y_train == 1], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")
		except IndexError:
			print('Handling the IndexError Exception...')
			ax.scatter(x[(y_train == 0)[:, 0]], y[(y_train == 0)[:, 0]], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
			ax.scatter(x[(y_train == 1)[:, 0]], y[(y_train == 1)[:, 0]], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")
		
		if labels:
			ax.set_xlabel(f'Feature {feats_plot[0]}')
			ax.set_ylabel(f'Feature {feats_plot[1]}')
		
		ax.legend()

		if save:
			plt.savefig(pwd + '/Local_Importance_Scoremap_{}.pdf'.format(name), bbox_inches='tight')
		else: 
			fig,ax=None,None

		return fig, ax

	def plot_complete_scoremap(self,name:str,dim:int,X: pd.DataFrame, y: np.array, pwd:str =os.getcwd()):
			"""Produce the Complete Local Feature Importance Scoremap: a Scoremap for each pair of features in the input dataset.   
			
			Parameters
			----------
			name: Dataset's name
			dim: Number of input features in the dataset
			X: Input dataset 
			y: Dataset labels
			pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
			
			Returns
			----------
			fig,ax : plt.figure  and plt.axes objects used to create the plot  
			"""
				
			fig, ax = plt.subplots(dim, dim, figsize=(50, 50))
			for i in range(dim):
				for j in range(i+1,dim):
						features = [i,j]
						# One of the successive two lines can be commented so that we obtain only one "half" of the 
						#matrix of plots to reduce a little bit the execution time. 
						_,_=self.plot_importance_map(name,X, y, 50, pwd, feats_plot = (features[0],features[1]), ax=ax[i,j],save=False,labels=False)
						_,_=self.plot_importance_map(name,X, y, 50, pwd, feats_plot = (features[1],features[0]), ax=ax[j,i],save=False,labels=False)

			plt.savefig(pwd+'/Local_Importance_Scoremap_{}_complete.pdf'.format(name),bbox_inches='tight')
			return fig,ax
