# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import numpy as np 
import pandas as pd
import pickle
import scipy

# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.iforest import IForest
from pyod.utils.data import generate_data


class TestIForest(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = IForest(contamination=self.contamination, random_state=42)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)
        assert (hasattr(self.clf, '_mu') and
                self.clf._mu is not None)
        assert (hasattr(self.clf, '_sigma') and
                self.clf._sigma is not None)
        assert (hasattr(self.clf, 'estimators_') and
                self.clf.estimators_ is not None)
        assert (hasattr(self.clf, 'estimators_samples_') and
                self.clf.estimators_samples_ is not None)
        assert (hasattr(self.clf, 'max_samples_') and
                self.clf.max_samples_ is not None)
        assert (hasattr(self.clf, '_max_features') and
                self.clf._max_features is not None)
        assert (hasattr(self.clf, 'estimators_features_') and
                self.clf.estimators_features_ is not None)
        assert (hasattr(self.clf, 'n_features_in_') and
                self.clf.n_features_in_ is not None)
        assert (hasattr(self.clf, 'offset_') and
                self.clf.offset_ is not None)
        

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(self.X_test,
                                                        method='linear',
                                                        return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    def test_predict_rank(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=3)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=3)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def test_feature_importances(self):
        feature_importances = self.clf.feature_importances_
        assert (len(feature_importances) == 2)

    # New tests inserted from here

    def test_decision_function_single_tree(self):

        X_train = np.array([[1, 1], [1, 2], [2, 1]])
        clf1 = IForest(contamination=0.1).fit(X_train)
        clf2 = IForest().fit(X_train)
        X=np.array([[2.0, 2.0]])
        tree_idx=np.random.randint(0,len(clf1.estimators_))

        assert_array_equal(
            clf1.decision_function_single_tree(tree_idx,X),
            clf1._score_samples(tree_idx,X) - clf1.offset_,
        )
        assert_array_equal(
            clf2.decision_function_single_tree(tree_idx,X),
            clf2._score_samples(tree_idx,X) - clf2.offset_,
        )

        #The decision function values could not be equal because clf1 and clf2 have 
        #two different contamination values. 

        assert_array_almost_equal(
            clf1.decision_function_single_tree(tree_idx,X), clf2.decision_function_single_tree(tree_idx,X),
            decimal=1
        )

        #Check weather the two decision function values are different

        assert not np.array_equal(clf1.decision_function_single_tree(tree_idx,X), clf2.decision_function_single_tree(tree_idx,X))

    def test_score_samples(self):

        X_train = np.array([[1, 1], [1, 2], [2, 1]])
        clf1 = IForest(contamination=0.1).fit(X_train)
        clf2 = IForest().fit(X_train)
        X=np.array([[2.0, 2.0]])
        tree_idx=np.random.randint(0,len(clf1.estimators_))
        assert_array_equal(
            clf1._score_samples(tree_idx,X),
            clf1.decision_function_single_tree(tree_idx,X) + clf1.offset_,
        )
        assert_array_equal(
            clf1._score_samples(tree_idx,X),
            clf2.decision_function_single_tree(tree_idx,X) + clf2.offset_,
        )
        assert_array_equal(
            clf1._score_samples(tree_idx,X), clf2._score_samples(tree_idx,X)
        )

    def test_compute_chunked_score_samples(self):

        X_train = np.array([[1, 1], [1, 2], [2, 1]])
        clf1 = IForest(contamination=0.1).fit(X_train)
        clf2 = IForest().fit(X_train)
        X=np.array([[2.0, 2.0]])
        tree_idx=np.random.randint(0,len(clf1.estimators_))

        assert not np.array_equal(
            clf1._compute_chunked_score_samples(tree_idx,X),
            clf1.decision_function_single_tree(tree_idx,X) + clf1.offset_,
        )
        assert not np.array_equal(
            clf2._compute_chunked_score_samples(tree_idx,X),
            clf2.decision_function_single_tree(tree_idx,X) + clf2.offset_,
        )
        
        assert_array_equal(
            clf1._compute_chunked_score_samples(tree_idx,X), clf2._compute_chunked_score_samples(tree_idx,X)
        )

    def test_compute_score_samples_single_tree(self):

        X_train = np.array([[1, 1], [1, 2], [2, 1]])
        clf1 = IForest(contamination=0.1).fit(X_train)
        clf2 = IForest().fit(X_train)
        X=np.array([[2.0, 2.0]])
        tree_idx=np.random.randint(0,len(clf1.estimators_))
        subsample_features=np.random.choice([True, False], size=1)

        assert not np.array_equal(
            clf1._compute_score_samples_single_tree(tree_idx,X,subsample_features),
            clf1.decision_function_single_tree(tree_idx,X) + clf1.offset_,
        )
        assert not np.array_equal(
            clf2._compute_score_samples_single_tree(tree_idx,X,subsample_features),
            clf2.decision_function_single_tree(tree_idx,X) + clf2.offset_,
        )

        assert_array_equal(
            clf1._compute_score_samples_single_tree(tree_idx,X,subsample_features), clf2._compute_score_samples_single_tree(tree_idx,X,subsample_features)
        )

    def test_diffi_ib(self):
        # create a random dataset
        np.random.seed(0)
        X = np.random.randn(100, 10)
        # create an isolation forest model
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X)
        # run the diffi_ib function
        fi_ib, exec_time = iforest.diffi_ib(X)
        #Check that all the elements of fi_ib are finite
        assert np.all(np.isfinite(fi_ib)) == True 
        # check that the output has the correct shape
        assert fi_ib.shape[0] == X.shape[1]
        # check that the execuiton time is positive
        assert exec_time > 0 

    def test_get_iic(self):
        # create a random dataset
        np.random.seed(0)
        X = np.random.randn(100, 10)
        # create an isolation forest model
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X)
        estimator=iforest.estimators_[np.random.randint(0,iforest.n_estimators)]
        is_leaves=np.random.choice([True, False], size=X.shape[0])
        adjust_iic=np.random.choice([True, False], size=1)
        lambda_outliers_ib = iforest._get_iic(estimator, X, is_leaves, adjust_iic=adjust_iic)

        assert type(lambda_outliers_ib) == np.ndarray
        assert lambda_outliers_ib.shape[0] == estimator.tree_.node_count
        assert np.all(lambda_outliers_ib >= -1) == True 

    def test_local_diffi(self):
        # create a random dataset
        np.random.seed(0)
        X = np.random.randn(100,10)
        # create an isolation forest model
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X)
        #Select a single sample from X at random
        x=X[np.random.randint(0,X.shape[0]),:]
        fi_ib, exec_time = iforest.local_diffi(x)

        assert np.all(np.isfinite(fi_ib)) == True
        assert fi_ib.shape[0] == x.shape[0]
        assert exec_time >= 0

    def test_local_diffi_batch(self):
        np.random.seed(0)
        X = np.random.randn(100,10)
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X)

        fi_ib,ord_idx,exec_time=iforest.local_diffi_batch(X)

        assert np.all(np.isfinite(fi_ib)) == True
        assert fi_ib.shape[0] == X.shape[0]
        assert ord_idx.shape == X.shape
        # Every element in ord_idx must be between 0 and X.shape[0]-1
        assert np.all(ord_idx >= X.shape[1]) == False 
        assert type(exec_time)==list
        assert np.all(np.array(exec_time)>=0) == True

    def test_compute_local_importances(self):
    
        #Create a path to save the pkl files created by compute_local_importances
        test_imp_score_path=os.path.join(os.getcwd(),'test_data','test_imp_score_local')
        test_plt_data_path=os.path.join(os.getcwd(),'test_data','test_plt_data_local')
        name='test_local_pima'

        #If the folder do not exist create them:
        if not os.path.exists(test_imp_score_path):
            os.makedirs(test_imp_score_path)
        if not os.path.exists(test_plt_data_path):
            os.makedirs(test_plt_data_path)

        # We will use the data contained in pima.mat for the tests
        path = os.path.join(os.getcwd(), 'data', 'ufs', 'pima.mat')
        data = scipy.io.loadmat(path)
        X_tr=data['X']
        y_tr=data['y']
        X, _ = shuffle(X_tr, y_tr, random_state=0)

        # create an isolation forest model
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X)

        fi,plt_data,path_fi,path_plt_data=iforest.compute_local_importances(X,name,pwd_imp_score=test_imp_score_path,pwd_plt_data=test_plt_data_path)

        """
        Tests on the pkl files
        """
        #Check that the returned path are strings
        assert type(path_fi) == str
        assert type(path_plt_data) == str
        #Check that the pkl files have been created
        assert os.path.exists(path_fi) == True
        assert os.path.exists(path_plt_data) == True
        #Check that the pkl files are not empty
        assert os.path.getsize(path_fi) > 0
        assert os.path.getsize(path_plt_data) > 0
        #Check that the pkl files can be loaded
        assert pickle.load(open(path_fi,'rb')) is not None
        assert pickle.load(open(path_plt_data,'rb')) is not None

        """
        Tests on fi and plt_data
        """
        #Check that all the elements of fi are finite
        assert np.all(np.isfinite(fi)) == True
        # check that the output has the correct shape
        assert fi.shape[0] == X.shape[0]
        #Extract the keys of plt_data
        plt_data_keys=list(plt_data.keys())
        imp,feat_ord,std=plt_data[plt_data_keys[0]],plt_data[plt_data_keys[1]],plt_data[plt_data_keys[2]]
        #Check that all the elements of imp are finite
        assert np.all(np.isfinite(imp)) == True
        #Check that the size of imp is correct
        assert imp.shape[0] == X.shape[1]
        #Check that the size of feat_ord is correct
        assert feat_ord.shape[0] == X.shape[1]
        #Values in feat_ord cannot be greater than X.shape[1]
        assert np.all(feat_ord>=X.shape[1]) == False
        #Check that the size of std is correct
        assert std.shape[0] == X.shape[1]
        #Check that all the elements of std are positive (standard deviation cannot be negative)
        assert np.all(std>=0) == True

    def test_compute_global_importances(self):

        #Create a path to save the pkl files created by compute_local_importances
        test_imp_score_path=os.path.join(os.getcwd(),'test_data','test_imp_score_global')
        test_plt_data_path=os.path.join(os.getcwd(),'test_data','test_plt_data_global')
        name='test_global_pima'

        #If the folder do not exist create them:
        if not os.path.exists(test_imp_score_path):
            os.makedirs(test_imp_score_path)
        if not os.path.exists(test_plt_data_path):
            os.makedirs(test_plt_data_path)

        # We will use the data contained in pima.mat for the tests
        path = os.path.join(os.getcwd(), 'data', 'ufs', 'pima.mat')
        data = scipy.io.loadmat(path)
        X_tr=data['X']
        y_tr=data['y']
        X, _ = shuffle(X_tr, y_tr, random_state=0)

        # create an isolation forest model
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X)
        nruns=np.random.randint(1,10)

        fi,plt_data,path_fi,path_plt_data=iforest.compute_global_importances(X,nruns,name,pwd_imp_score=test_imp_score_path,pwd_plt_data=test_plt_data_path)

        """
        Tests on the pkl files
        """

        #Check that the returned path are strings
        assert type(path_fi) == str
        assert type(path_plt_data) == str
        #Check that the pkl files have been created
        assert os.path.exists(path_fi) == True
        assert os.path.exists(path_plt_data) == True
        #Check that the pkl files are not empty
        assert os.path.getsize(path_fi) > 0
        assert os.path.getsize(path_plt_data) > 0
        #Check that the pkl files can be loaded
        assert pickle.load(open(path_fi,'rb')) is not None
        assert pickle.load(open(path_plt_data,'rb')) is not None

        """
        Tests on fi and plt_data
        """
        #Check that nruns is positive 
        assert nruns >= 0
        #Check that all the elements of fi are finite
        assert np.all(np.isfinite(fi)) == True
        # check that the output has the correct shape
        assert fi.shape[1] == X.shape[1]
        #Extract the keys of plt_data
        plt_data_keys=list(plt_data.keys())
        imp,feat_ord,std=plt_data[plt_data_keys[0]],plt_data[plt_data_keys[1]],plt_data[plt_data_keys[2]]
        #Check that all the elements of imp are finite
        assert np.all(np.isfinite(imp)) == True
        #Check that the size of imp is correct
        assert imp.shape[0] == X.shape[1]
        #Check that the size of feat_ord is correct
        assert feat_ord.shape[0] == X.shape[1]
        #Values in feat_ord cannot be greater than X.shape[1]
        assert np.all(feat_ord>=X.shape[1]) == False
        #Check that the size of std is correct
        assert std.shape[0] == X.shape[1]
        #Check that all the elements of std are positive (standard deviation cannot be negative)
        assert np.all(std>=0) == True

    def test_plot_importances_bars(self):

        # We need a feature importance 2d array with the importance values. 
        # We can extract it from the pkl files created by the test_compure_global_importances 
        # and test_compute_local_importances functions

        #We create the plot with plot_importances_bars and we will then compare it with the 
        #expected result contained in GFI_glass_synt.pdf
        imps_path=os.path.join(os.getcwd(),'test_data','test_imp_score_global','imp_score_LFI_test_global_pima.pkl')

        imps=pickle.load(open(imps_path,'rb'))

        #Create a path to save the plot image 
        plot_path=os.path.join(os.getcwd(),'test_data','test_plots')

        #If the folder do not exist create it:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        #Create an IForest object to call the plt_importances_bars method
        iforest=IForest()

        #Create a name for the plot
        name='test_pima'
        f=6
        fig,ax,bars=iforest.plt_importances_bars(imps_path,name,pwd=plot_path,f=f)

        """
        Tests on ax
        """
        #Check that the returned ax is not None
        assert ax is not None
        assert fig is not None
        #Check that the returned ax is an axis object
        #assert type(ax) == matplotlib.axes._subplots.AxesSubplot
        #Check that the x label is correct
        assert ax.get_xlabel() == 'Rank'
        #Check that the y label is correct
        assert ax.get_ylabel() == 'Percentage count'
        #Check that the xtick  and y tick labels are correct
        x_tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        y_tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert x_tick_labels == ['$1^{st}$', '$2^{nd}$', '$3^{rd}$', '$4^{th}$', '$5^{th}$', '$6^{th}$']
        assert y_tick_labels == ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

        #See if the plot correctly changes if I pass from f=6 (default value) to f=9
        f1=9
        fig1,ax1,bars1=iforest.plt_importances_bars(imps_path,name='test_pima_9',pwd=plot_path,f=f1)

        #Check that the xtick  and y tick labels are correct
        x_tick_labels1 = [tick.get_text() for tick in ax1.get_xticklabels()]
        assert x_tick_labels1 == ['$1^{st}$', '$2^{nd}$', '$3^{rd}$', '$4^{th}$', '$5^{th}$', '$6^{th}$','$7^{th}$','$8^{th}$','$9^{th}$']

        """
        Tests on bars

        The main test o perform on bars is that the sum of the percentages values on each column should be 100. 
        """
        assert type(bars) == pd.DataFrame
        assert bars.shape == (imps.shape[1],imps.shape[1])
        assert np.all(bars.sum()==100) == True
        #Same on bars1
        assert type(bars1) == pd.DataFrame
        assert bars1.shape == (imps.shape[1],imps.shape[1])
        assert np.all(bars1.sum()==100) == True

    def test_plt_feat_bar_plot(self):

        # We need the plt_data array: let's consider the global case with plt_data_GFI_glass.pkl and 
        # the local case with plt_data_LFI_glass.pkl

        plt_data_global_path=os.path.join(os.getcwd(),'plt_data','plt_data_GFI_test_global_pima.pkl')
        plt_data_local_path=os.path.join(os.getcwd(),'plt_data','plt_data_LFI_test_local_pima.pkl')

        name_global='test_GFI_pima'
        name_local='test_LFI_pima'

        plot_path=os.path.join(os.getcwd(),'test_data','test_plots')

        #Create an IForest object to call the plt_feat_bar_plot method
        iforest=IForest()

        ax1,ax2=iforest.plt_feat_bar_plot(plt_data_global_path,name_global,pwd=plot_path,is_local=False)
        ax3,ax4=iforest.plt_feat_bar_plot(plt_data_local_path,name_local,pwd=plot_path,is_local=True)

        y_tick_labels_local = [tick.get_text() for tick in ax3.get_yticklabels()]
        y_tick_labels2_local = [tick.get_text() for tick in ax4.get_yticklabels()]
        y_tick_labels_global = [tick.get_text() for tick in ax1.get_yticklabels()]
        y_tick_labels2_global = [tick.get_text() for tick in ax2.get_yticklabels()]

        """
        Tests on ax1,ax2,ax3,ax4
        """
        #Check that the returned ax is not None
        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None
        assert ax4 is not None
        #Check that the x label is correct
        assert ax1.get_xlabel() == 'Importance Score'
        #Check that the y label is correct
        assert ax1.get_ylabel() == 'Features'
        #Check that the x label is correct
        assert ax3.get_xlabel() == 'Importance Score'
        #Check that the y label is correct
        assert ax3.get_ylabel() == 'Features'
        #Check that the xtick  and ytick labels are correct
        assert np.all(np.array(y_tick_labels_local).astype('float')>=len(y_tick_labels2_local)-1) == False
        assert np.all(np.array(y_tick_labels_global).astype('float')>=len(y_tick_labels2_global)-1) == False

    def test_plot_importance_map(self):

        # Let's perform the test on the pima.mat dataset 
        path = os.path.join(os.getcwd(), 'data', 'ufs', 'pima.mat')
        data = scipy.io.loadmat(path)
        X_tr=data['X']
        y_tr=data['y']
        X, y = shuffle(X_tr, y_tr, random_state=0)

        name='test_pima'

        # create an isolation forest model
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X_tr)
        plot_path=os.path.join(os.getcwd(),'test_data','test_plots')

        fig,ax=iforest.plot_importance_map(name,X,y,30,pwd=plot_path)

        """
        Tests on ax
        """

        #Check that the returned ax is not None
        assert ax is not None
        assert fig is not None

    def test_plot_complete_scoremap(self):
        
        # Here we'll use a random dataset with just 3 features otherwise it takes too much time to 
        #create the plots
        np.random.seed(0)
        X = np.random.randn(100, 3)
        #Assign at random the anomalous/not anomaoous labels
        #Create a random array of 0 and 1 of shape=(100,)
        y=np.random.randint(0,2,size=100)
        name='test_complete'
        # create an isolation forest model
        iforest = IForest(n_estimators=10, max_samples=64, random_state=0)
        iforest.fit(X)
        plot_path=os.path.join(os.getcwd(),'test_data','test_plots')

        fig,ax=iforest.plot_complete_scoremap(name,X.shape[1],iforest,X,y,pwd=plot_path)
            
        """
        Tests on ax
        """

        #Check that the returned ax is not None
        assert ax is not None
        assert fig is not None

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
