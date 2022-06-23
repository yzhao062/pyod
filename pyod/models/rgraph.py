# -*- coding: utf-8 -*-
""" R-graph

"""
# Author: Michiel Bongaerts (but not author of the R-graph method)
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..utils.utility import check_parameter

from .base import BaseDetector
from .base_dl import _get_tensorflow_version



from scipy import sparse
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
import warnings

class RGraph(BaseDetector):

    def __init__(self, transition_steps = 10, n_nonzero = 10 , gamma = 50.0, gamma_nz = True,
                algorithm = 'lasso_lars', tau = 1.0, preprocessing = True, contamination = 0.1,
                support_init='L2', maxiter = 40, support_size= 100, active_support = True, verbose = True):

        super(RGraph, self).__init__(contamination = contamination)

        self.transition_steps = transition_steps
        self.n_nonzero = n_nonzero
        self.gamma =gamma
        self.gamma_nz = gamma_nz
        self.algorithm = algorithm
        self.tau = tau
        self.preprocessing = preprocessing
        self.contamination = contamination
        self.support_init = support_init
        self.maxiter = maxiter
        self.support_size = support_size
        self.active_support = active_support
        self.verbose = verbose


    

        
        
    
    def active_support_elastic_net(self, X, y, alpha, tau=1.0, algorithm='lasso_lars', support_init='L2', 
                                   support_size=100, maxiter=40):
        """
        Source: https://github.com/ChongYou/subspace-clustering/blob/master/cluster/selfrepresentation.py
            An active support based algorithm for solving the elastic net optimization problem
            min_{c} tau ||c||_1 + (1-tau)/2 ||c||_2^2 + alpha / 2 ||y - c X ||_2^2.
        
        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (1, n_features)
        alpha : float
        tau : float, default 1.0
        algorithm : string, default ``spams``
            Algorithm for computing solving the subproblems. Either lasso_lars or lasso_cd or spams
            (installation of spams package is required).
            Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
        support_init: string, default ``knn``
            This determines how the active support is initialized.
            It can be either ``knn`` or ``L2``.
        support_size: int, default 100
            This determines the size of the working set.
            A small support_size decreases the runtime per iteration while increase the number of iterations.
        maxiter: int default 40
            Termination condition for active support update.
        
        Returns
        -------
        c : shape n_samples
            The optimal solution to the optimization problem.
        """
        n_samples = X.shape[0]

        if n_samples <= support_size:  # skip active support search for small scale data
            supp = np.arange(n_samples, dtype=int)  # this results in the following iteration to converge in 1 iteration
        else:    
            if support_init == 'L2':
                L2sol = np.linalg.solve(np.identity(y.shape[1]) * alpha + np.dot(X.T, X), y.T)
                c0 = np.dot(X, L2sol)[:, 0]
                supp = np.argpartition(-np.abs(c0), support_size)[0:support_size]
            elif support_init == 'knn':
                supp = np.argpartition(-np.abs(np.dot(y, X.T)[0]), support_size)[0:support_size]

        curr_obj = float("inf")
        for _ in range(maxiter):
            Xs = X[supp, :]
            
            ### Removed the original option to use 'spams' since this would need the spams dependency 
            # if algorithm == 'spams':
            #     cs = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(Xs.T), 
            #                      lambda1=tau*alpha, lambda2=(1.0-tau)*alpha)
            #     cs = np.asarray(cs.todense()).T
            # else:
            cs = sparse_encode(y, Xs, algorithm=algorithm, alpha=alpha)
          
            delta = (y - np.dot(cs, Xs)) / alpha

            obj = tau * np.sum(np.abs(cs[0])) + (1.0 - tau)/2.0 * np.sum(np.power(cs[0], 2.0)) + alpha/2.0 * np.sum(np.power(delta, 2.0))
            if curr_obj - obj < 1.0e-10 * curr_obj:
                break
            curr_obj = obj

            coherence = np.abs(np.dot(delta, X.T))[0]
            coherence[supp] = 0
            addedsupp = np.nonzero(coherence > tau + 1.0e-10)[0]
            
            if addedsupp.size == 0:  # converged
                break

            # Find the set of nonzero entries of cs.
            activesupp = supp[np.abs(cs[0]) > 1.0e-10]  
            
            if activesupp.size > 0.8 * support_size:  # this suggests that support_size is too small and needs to be increased
                support_size = min([round(max([activesupp.size, support_size]) * 1.1), n_samples])
            
            if addedsupp.size + activesupp.size > support_size:
                ord = np.argpartition(-coherence[addedsupp], support_size - activesupp.size)[0:support_size - activesupp.size]
                addedsupp = addedsupp[ord]
            
            supp = np.concatenate([activesupp, addedsupp])
        
        c = np.zeros(n_samples)
        c[supp] = cs
        return c



    def elastic_net_subspace_clustering(self, X, gamma=50.0, gamma_nz=True, tau=1.0, algorithm='lasso_lars', 
                                        active_support=True, active_support_params=None, n_nonzero=50 ):
        """
        Source: https://github.com/ChongYou/subspace-clustering/blob/master/cluster/selfrepresentation.py
        
        Elastic net subspace clustering (EnSC) [1]. 
        Compute self-representation matrix C from solving the following optimization problem
        min_{c_j} tau ||c_j||_1 + (1-tau)/2 ||c_j||_2^2 + alpha / 2 ||x_j - c_j X ||_2^2 s.t. c_jj = 0,
        where c_j and x_j are the j-th rows of C and X, respectively.
        
        Parameter ``algorithm`` specifies the algorithm for solving the optimization problem.
        ``lasso_lars`` and ``lasso_cd`` are algorithms implemented in sklearn, 
        ``spams`` refers to the same algorithm as ``lasso_lars`` but is implemented in 
        spams package available at http://spams-devel.gforge.inria.fr/ (installation required)
        In principle, all three algorithms give the same result.    
        For large scale data (e.g. with > 5000 data points), use any of these algorithms in
        conjunction with ``active_support=True``. It adopts an efficient active support 
        strategy that solves the optimization problem by breaking it into a sequence of 
        small scale optimization problems as described in [1].
        If tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
        If tau = 0.0, the method reduces to least squares regression (LSR) [3].
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data to be clustered
        gamma : float
        gamma_nz : boolean, default True
            gamma and gamma_nz together determines the parameter alpha. When ``gamma_nz = False``, 
            alpha = gamma. When ``gamma_nz = True``, then alpha = gamma * alpha0, where alpha0 is 
            the largest number such that the solution to the optimization problem with alpha = alpha0
            is the zero vector (see Proposition 1 in [1]). Therefore, when ``gamma_nz = True``, gamma
            should be a value greater than 1.0. A good choice is typically in the range [5, 500].   
        tau : float, default 1.0
            Parameter for elastic net penalty term. 
            When tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
            When tau = 0.0, the method reduces to least squares regression (LSR) [3].
        algorithm : string, default ``lasso_lars``
            Algorithm for computing the representation. Either lasso_lars or lasso_cd or spams 
            (installation of spams package is required).
            Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
        n_nonzero : int, default 50
            This is an upper bound on the number of nonzero entries of each representation vector. 
            If there are more than n_nonzero nonzero entries,  only the top n_nonzero number of
            entries with largest absolute value are kept.
        active_support: boolean, default True
            Set to True to use the active support algorithm in [1] for solving the optimization problem.
            This should significantly reduce the running time when n_samples is large.
        active_support_params: dictionary of string to any, optional
            Parameters (keyword arguments) and values for the active support algorithm. It may be
            used to set the parameters ``support_init``, ``support_size`` and ``maxiter``, see
            ``active_support_elastic_net`` for details. 
            Example: active_support_params={'support_size':50, 'maxiter':100}
            Ignored when ``active_support=False``
        
        Returns
        -------
        representation_matrix_ : csr matrix, shape: n_samples by n_samples
            The self-representation matrix.
        
        References
        ----------- 
        [1] C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
        [2] E. Elhaifar, R. Vidal, Sparse Subspace Clustering: Algorithm, Theory, and Applications, TPAMI 2013
        [3] C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
        """


        if( (algorithm in ('lasso_lars', 'lasso_cd')) and (tau < 1.0 - 1.0e-10) ) :  
            warnings.warn('algorithm {} cannot handle tau smaller than 1. Using tau = 1'.format(algorithm))
            tau = 1.0

        if active_support == True and active_support_params == None:
            active_support_params = {}

        n_samples = X.shape[0]
        rows = np.zeros(n_samples * n_nonzero)
        cols = np.zeros(n_samples * n_nonzero)
        vals = np.zeros(n_samples * n_nonzero)
        curr_pos = 0
     
        gamma_is_zero_notification = False
        for i in range(n_samples):
            if( (i % 25 == 0) and (self.verbose == 1) ):
                print('{}/{}'.format(i,n_samples))

            y = X[i, :].copy().reshape(1, -1)
            X[i, :] = 0
            
            if algorithm in ('lasso_lars', 'lasso_cd'):
                if gamma_nz == True:
                    coh = np.delete(np.absolute(np.dot(X, y.T)), i)
                    alpha0 = np.amax(coh) / tau  # value for which the solution is zero
                    alpha = alpha0 / gamma
                else:
                    alpha = 1.0 / gamma
                    
                if( gamma > 10**4):
                    if( gamma_is_zero_notification == False):
                        print('Set alpha = 0 i.e. LinearRegression() is used')
                        gamma_is_zero_notification = True

                    alpha = 0
                    

                if( alpha == 0):
                    lr = LinearRegression()
                    lr.fit(X.T , y[0]  )
                    c = lr.coef_
                        

                elif active_support == True:
                    c = self.active_support_elastic_net(X, y, alpha, tau, algorithm, **active_support_params)
                else:
                    # if algorithm == 'spams':
                        # c = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(X.T), 
                        #                 lambda1=tau * alpha, lambda2=(1.0-tau) * alpha)
                        # c = np.asarray(c.todense()).T[0]
                    # else:
                    c = sparse_encode(y, X, algorithm=algorithm, alpha=alpha)[0]

            else:
                warnings.warn("algorithm {} not found".format(algorithm))

            index = np.flatnonzero(c)
            if index.size > n_nonzero:
            #  warnings.warn("The number of nonzero entries in sparse subspace clustering exceeds n_nonzero")
                index = index[np.argsort(-np.absolute(c[index]))[0:n_nonzero]]
            rows[curr_pos:curr_pos + len(index)] = i
            cols[curr_pos:curr_pos + len(index)] = index
            vals[curr_pos:curr_pos + len(index)] = c[index]
            curr_pos += len(index)
            
            X[i, :] = y

        return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))





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
        X = check_array(X)
        self._set_n_classes(y)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
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

        X = check_array(X)
        
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)


        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)


        A = self.elastic_net_subspace_clustering(X_norm, gamma = self.gamma , gamma_nz = self.gamma_nz, 
                                                 tau= self.tau, algorithm= self.algorithm, 
                                                 active_support= self.active_support, n_nonzero = self.n_nonzero,
                                                 active_support_params={'support_init' : self.support_init, 'support_size': self.support_size, 'maxiter':self.maxiter}
                                                 )


        self.P = normalize( np.abs(  A.toarray() ), norm='l1')

        pi = np.ones((1, len(self.P)), dtype = 'float64' ) / len(self.P)
        pi_bar = np.zeros((1, len(self.P)), dtype = 'float64')

        # Do transition steps
        for _ in range(self.transition_steps):
            pi = pi @ self.P
            pi_bar += pi

        pi_bar /= self.transition_steps
        scores = pi_bar[0]
    
        # smaller scores correspond with outliers, thus we use 1- score such that
        # higher scores are associated with outliers
        scores = -1 * scores

        if hasattr(self, 'X_train'):
            scores = scores[-original_size:]

        return scores
    
    
    
    


