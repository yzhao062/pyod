All Models
==========

All models inherit from :class:`pyod.models.base.BaseDetector`, which provides the shared API: ``fit``, ``predict``, ``decision_function``, ``predict_proba``, ``predict_confidence``, and ``predict_with_rejection``. See :doc:`api_cc` for the full list. The pages below document only each model's own parameters and methods.


pyod.models.abod module
-----------------------

.. automodule:: pyod.models.abod
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:




pyod.models.ae1svm module
-------------------------

.. automodule:: pyod.models.ae1svm
    :members:
    :exclude-members: get_params, set_params, RandomFourierFeatures, InnerAE1SVM, _train_autoencoder
    :undoc-members:
    :show-inheritance:




pyod.models.alad module
-----------------------

.. automodule:: pyod.models.alad
    :members:
    :exclude-members: get_params, set_params, train_step, train_more, get_outlier_scores
    :undoc-members:
    :show-inheritance:



pyod.models.anogan module
-------------------------

.. automodule:: pyod.models.anogan
    :members:
    :exclude-members: Discriminator, Generator, QueryModel
    :undoc-members:
    :show-inheritance:



pyod.models.auto\_encoder module
--------------------------------

.. automodule:: pyod.models.auto_encoder
    :members:
    :exclude-members: get_params, set_params, decision_function_update, epoch_update, evaluating_prepare, evaluating_forward, AutoEncoderModel, load
    :undoc-members:
    :show-inheritance:



pyod.models.auto\_encoder\_torch module
---------------------------------------

.. automodule:: pyod.models.auto_encoder_torch
    :members:
    :exclude-members: get_params, set_params, InnerAutoencoder
    :exclude-members:
    :show-inheritance:



pyod.models.cblof module
------------------------

.. automodule:: pyod.models.cblof
    :members:
    :exclude-members: get_params, set_params,
    :undoc-members:
    :show-inheritance:



pyod.models.cof module
----------------------

.. automodule:: pyod.models.cof
    :members:
    :exclude-members: get_params, set_params,
    :undoc-members:
    :show-inheritance:



pyod.models.combination module
------------------------------

.. automodule:: pyod.models.combination
    :members:
    :exclude-members: get_params, set_params,
    :undoc-members:
    :show-inheritance:



pyod.models.cd module
---------------------

.. automodule:: pyod.models.cd
    :members:
    :exclude-members: get_params, set_params,
    :undoc-members:
    :show-inheritance:



pyod.models.copod module
------------------------

.. automodule:: pyod.models.copod
    :members:
    :exclude-members:
    :undoc-members:
    :show-inheritance:



pyod.models.deep\_svdd module
-----------------------------

.. automodule:: pyod.models.deep_svdd
    :members:
    :exclude-members: InnerDeepSVDD
    :undoc-members:
    :show-inheritance:



pyod.models.devnet module
-----------------------------

.. automodule:: pyod.models.devnet
    :members:
    :exclude-members: DevNetD, DevNetS, DevNetLinear, deviation_loss, train_and_test, deviation_network, SupDataset, input_batch_generation_sup_sparse, load_model_weight_predict
    :undoc-members:
    :show-inheritance:



pyod.models.dif module
-----------------------------

.. automodule:: pyod.models.dif
    :members:
    :exclude-members: LinearBlock, MLPnet
    :undoc-members:
    :show-inheritance:




pyod.models.embedding module
-----------------------------

.. automodule:: pyod.models.embedding
    :members:
    :exclude-members: get_params, set_params, resolve_detector, _DETECTOR_SHORTCUTS
    :undoc-members:
    :show-inheritance:



pyod.models.ecod module
------------------------

.. automodule:: pyod.models.ecod
    :members:
    :exclude-members: InnerDeepSVDD,
    :undoc-members:
    :show-inheritance:



pyod.models.feature\_bagging module
-----------------------------------

.. automodule:: pyod.models.feature_bagging
    :members:
    :undoc-members:
    :show-inheritance:




pyod.models.gmm module
----------------------

.. automodule:: pyod.models.gmm
    :members:
    :exclude-members: weights_, get_params, set_params, n_iter_, lower_bound_, means_, converged_, covariances_
    :undoc-members:
    :show-inheritance:



pyod.models.hbos module
-----------------------

.. automodule:: pyod.models.hbos
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.hdbscan module
--------------------------

.. automodule:: pyod.models.hdbscan
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.iforest module
--------------------------

.. automodule:: pyod.models.iforest
    :members:
    :exclude-members: estimators_, estimators_features_, estimators_samples_, get_params, set_params,
    :undoc-members:
    :show-inheritance:



pyod.models.inne module
-----------------------

.. automodule:: pyod.models.inne
    :members:
    :exclude-members:
    :undoc-members:
    :show-inheritance:



pyod.models.kde module
----------------------

.. automodule:: pyod.models.kde
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.knn module
----------------------

.. automodule:: pyod.models.knn
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.kpca module
-----------------------

.. automodule:: pyod.models.kpca
    :members:
    :exclude-members: PyODKernelPCA
    :undoc-members:
    :show-inheritance:



pyod.models.lmdd module
-----------------------

.. automodule:: pyod.models.lmdd
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.loda module
-----------------------

.. automodule:: pyod.models.loda
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.lof module
----------------------

.. automodule:: pyod.models.lof
    :members:
    :exclude-members: negative_outlier_factor_, n_neighbors_
    :undoc-members:
    :show-inheritance:



pyod.models.loci module
-----------------------

.. automodule:: pyod.models.loci
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.lunar module
------------------------

.. automodule:: pyod.models.lunar
    :members:
    :exclude-members: SCORE_MODEL, WEIGHT_MODEL
    :show-inheritance:



pyod.models.lscp module
-----------------------

.. automodule:: pyod.models.lscp
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.mad module
----------------------

.. automodule:: pyod.models.mad
    :members:
    :exclude-members:
    :undoc-members:
    :show-inheritance:



pyod.models.mcd module
----------------------

.. automodule:: pyod.models.mcd
    :members:
    :exclude-members: raw_location_, raw_covariance_, raw_support_, location_, covariance_, precision_, support_
    :undoc-members:
    :show-inheritance:



pyod.models.mo\_gaal module
---------------------------

.. automodule:: pyod.models.mo_gaal
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.ocsvm module
------------------------

.. automodule:: pyod.models.ocsvm
    :members:
    :exclude-members: coef_, dual_coef_, support_, support_vectors_, intercept_
    :undoc-members:
    :show-inheritance:



pyod.models.pca module
----------------------

.. automodule:: pyod.models.pca
    :members:
    :exclude-members: components_, explained_variance_ratio_, singular_values_, mean_
    :undoc-members:
    :show-inheritance:



pyod.models.qmcd module
-------------------------

.. automodule:: pyod.models.qmcd
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.rgraph module
-------------------------

.. automodule:: pyod.models.rgraph
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.rod module
----------------------

.. automodule:: pyod.models.rod
    :members:
    :exclude-members: angle, euclidean, geometric_median, mad, process_sub, rod_3D, rod_nD, scale_angles, sigmoid
    :undoc-members:
    :show-inheritance:



pyod.models.sampling module
---------------------------

.. automodule:: pyod.models.sampling
    :members:
    :undoc-members:
    :show-inheritance:




pyod.models.sod module
----------------------

.. automodule:: pyod.models.sod
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.so\_gaal module
---------------------------

.. automodule:: pyod.models.so_gaal
    :members:
    :exclude-members: Discriminator, Generator
    :undoc-members:
    :show-inheritance:



pyod.models.sos module
----------------------

.. automodule:: pyod.models.sos
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.suod module
-----------------------

.. automodule:: pyod.models.suod
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.thresholds module
-----------------------

.. automodule:: pyod.models.thresholds
    :members:
    :undoc-members:
    :show-inheritance:



pyod.models.vae module
----------------------

.. automodule:: pyod.models.vae
    :members:
    :exclude-members: VAEModel, vae_loss
    :undoc-members:
    :show-inheritance:



pyod.models.xgbod module
------------------------

.. automodule:: pyod.models.xgbod
    :members:
    :undoc-members:
    :show-inheritance:




Module contents
---------------

.. automodule:: pyod.models
    :members:
    :undoc-members:
    :show-inheritance:



.. rubric:: References

.. bibliography::
   :cited:
   :labelprefix: B
