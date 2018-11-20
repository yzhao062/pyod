Examples
========

.. toctree::

    examples


kNN Example
-----------

Full example: `knn_example.py <https://github.com/yzhao062/Pyod/blob/master/examples/knn_example.py>`_

1. Import models

    .. code-block:: python

        from pyod.models.knn import KNN   # kNN detector


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

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

4. Evaluate the prediction using ROC and Precision\@rank n :func:`pyod.utils.data.evaluate_print`:

    .. code-block:: python

        # evaluate and print the results
        print("\nOn Training Data:")
        evaluate_print(clf_name, y_train, y_train_scores)
        print("\nOn Test Data:")
        evaluate_print(clf_name, y_test, y_test_scores)

5. See sample outputs on both training and test data:

    .. code-block:: bash

        On Training Data:
        KNN ROC:1.0, precision @ rank n:1.0

        On Test Data:
        KNN ROC:0.9989, precision @ rank n:0.9

6. Generate the visualizations by visualize function included in all examples:

    .. code-block:: python

        visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
                  y_test_pred, show_figure=True, save_figure=False)


.. figure:: figs/KNN.png
    :alt: kNN demo


Model Combination Example
-------------------------
`comb_example.py <https://github.com/yzhao062/Pyod/blob/master/examples/comb_example.py>`_ is a quick demo for showing the API for combining multiple algorithms.
Given we have *n* individual outlier detectors, each of them generates an individual score for all samples. The task is to combine the outputs from these detectors effectivelly.

**Model combination example** is made available below
(`Code <https://github.com/yzhao062/Pyod/blob/master/examples/comb_example.py>`_, `Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/Pyod/master>`_):

For Jupyter Notebooks, please navigate to **"/notebooks/Model Combination.ipynb"**

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

5. Finally, all four combination methods are evaluated with ROC and Precision
   @ Rank n:

    .. code-block:: bash

        Combining 20 kNN detectors
        Combination by Average ROC:0.9194, precision @ rank n:0.4531
        Combination by Maximization ROC:0.9198, precision @ rank n:0.4688
        Combination by AOM ROC:0.9257, precision @ rank n:0.4844
        Combination by MOA ROC:0.9263, precision @ rank n:0.4688
