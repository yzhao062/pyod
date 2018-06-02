Examples
============

.. toctree::

    examples


kNN Example
----------------

Full example: `knn_example.py <https://github.com/yzhao062/Pyod/blob/master/examples/knn_example.py>`_

1. Import models

    .. code-block:: python

        from pyod.models.knn import KNN # kNN detector
        from pyod.utils.data import generate_data

2. Generate sample data with :func:`pyod.utils.data.generate_data`:

    .. code-block:: python

        contamination = 0.1  # percentage of outliers
        n_train = 200  # number of training points
        n_test = 100  # number of testing points

        X_train, y_train, X_test, y_test = generate_data(
            n_train=n_train, n_test=n_test, contamination=contamination)

3. Initialize a :class:`pyod.models.knn.KNN` detector, fit the model, and make
   the prediction:

    .. code-block:: python

        # train kNN detector
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(X_train)

        # get the prediction label and decision_scores_ on the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

4. Evaluate the prediction by ROC and Precision\@rank n :func:`pyod.utils.utility.precision_n_scores`:

    .. code-block:: python

        import numpy as np
        from sklearn.metrics import roc_auc_score
        from pyod.utils.utility import precision_n_scores

        # evaluate and print the results
        print('{clf_name} Train ROC:{roc}, precision @ rank n:{prn}'.format(
            clf_name=clf_name,
            roc=np.round(roc_auc_score(y_train, y_train_scores), decimals=4),
            prn=np.round(precision_n_scores(y_train, y_train_scores), decimals=4)))

        print('{clf_name} Test ROC:{roc}, precision @ rank n:{prn}'.format(
            clf_name=clf_name,
            roc=np.round(roc_auc_score(y_test, y_test_scores), decimals=4),
            prn=np.round(precision_n_scores(y_test, y_test_scores), decimals=4)))

5. See sample outputs on both training and test data:

    .. code-block:: bash

        KNN Train ROC:0.9676, precision @ rank n:0.95
        KNN Test ROC:1.0, precision @ rank n:1.0

6. Generate the visualizations by :func:`pyod.utils.data.visualize`

    .. code-block:: python

        from pyod.utils.data import visualize

        visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
          y_test_pred, save_figure=True)

.. figure:: figs/KNN.png
    :alt: kNN demo


Model Combination Example
--------------------------
`comb_example.py <https://github.com/yzhao062/Pyod/blob/master/examples/comb_example.py>`_ is a quick demo for showing the API for combining multiple algorithms.
Given we have *n* individual outlier detectors, each of them generates an individual score for all samples. The task is to combine the outputs from these detectors effectivelly.

1. Import models and generate sample data:

    .. code-block:: python

        from pyod.models.knn import KNN  # kNN detector
        from pyod.models.combination import aom, moa, average, maximization
        from pyod.utils.data import generate_data

        X, y= generate_data(train_only=True)  # load data


2. First initialize 20 kNN outlier detectors with different k (10 to 200), and get the outlier scores:

    .. code-block:: python

        # initialize 20 base detectors for combination
        k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                    150, 160, 170, 180, 190, 200]

        train_scores = np.zeros([X_train.shape[0], n_clf])
        test_scores = np.zeros([X_test.shape[0], n_clf])

        for i in range(n_clf):
            k = k_list[i]

            clf = KNN(n_neighbors=k, method='largest')
            clf.fit(X_train_norm)

            train_scores[:, i] = clf.decision_scores_
            test_scores[:, i] = clf.decision_function(X_test_norm)

3. Then the output codes are standardized into zero average and unit std before combination:

    .. code-block:: python

        from pyod.utils.utility import standardizer

        # scores have to be normalized before combination
        train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

4. Then four different combination algorithms are applied as described above:

    .. code-block:: python

        comb_by_average = average(test_scores_norm)
        comb_by_maximization = maximization(test_scores_norm)
        comb_by_aom = aom(test_scores_norm, 5) # 5 groups
        comb_by_moa = moa(test_scores_norm, 5)) # 5 groups

5. Finally, all four combination methods are evaluated with 20 iterations:

    .. code-block:: bash

        Combining 20 kNN detectors
        ite 1 comb by average, ROC: 0.9014 precision@n_train: 0.4531
        ite 1 comb by maximization, ROC: 0.9014 precision@n_train: 0.5
        ite 1 comb by aom, ROC: 0.9081 precision@n_train: 0.5
        ite 1 comb by moa, ROC: 0.9052 precision@n_train: 0.4843
        ...

        Summary of 10 iterations
        comb by average, ROC: 0.9196, precision@n: 0.5464
        comb by maximization, ROC: 0.9198, precision@n: 0.5532
        comb by aom, ROC: 0.9260, precision@n: 0.5630
        comb by moa, ROC: 0.9244, precision@n: 0.5523
