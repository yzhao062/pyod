Thresholding
============

Full example: `threshold_example.py <https://github.com/yzhao062/Pyod/blob/master/examples/threshold_example.py>`_

1. Import models

    .. code-block:: python

        from pyod.models.knn import KNN              # kNN detector
        from pyod.models.thresholds import FILTER    # Filter thresholder
        from pyod.utils.data import generate_data    # synthetic data generator


2. Generate sample data with :func:`pyod.utils.data.generate_data`:

    .. code-block:: python

        contamination = 0.1  # percentage of outliers
        n_train = 200  # number of training points
        n_test = 100  # number of testing points

        X_train, X_test, y_train, y_test = generate_data(
            n_train=n_train, n_test=n_test, contamination=contamination)

3. Initialize a :class:`pyod.models.knn.KNN` detector, fit the model, and make
   the prediction.

    .. code-block:: python

        # train kNN detector and apply FILTER thresholding
        clf_name = 'KNN'
        clf = KNN(contamination=FILTER())
        clf.fit(X_train)

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores
