Model Combination
=================

Outlier detection often suffers from model instability due to its unsupervised
nature. Thus, it is recommended to combine various detector outputs, e.g., by averaging,
to improve its robustness. Detector combination is a subfield of outlier ensembles;
refer :cite:`b-kalayci2018anomaly` for more information.


Four score combination mechanisms are shown in this demo:


#. **Average**: average scores of all detectors.
#. **maximization**: maximum score across all detectors.
#. **Average of Maximum (AOM)**: divide base detectors into subgroups and take the maximum score for each subgroup. The final score is the average of all subgroup scores.
#. **Maximum of Average (MOA)**: divide base detectors into subgroups and take the average score for each subgroup. The final score is the maximum of all subgroup scores.


"examples/comb_example.py" illustrates the API for combining the output of multiple base detectors
(\ `comb_example.py <https://github.com/yzhao062/pyod/blob/master/examples/comb_example.py>`_\ ,
`Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyod/master>`_\ ). For Jupyter Notebooks,
please navigate to **"/notebooks/Model Combination.ipynb"**


1. Import models and generate sample data.

    .. code-block:: python

        import numpy as np
        from pyod.models.knn import KNN  # kNN detector
        from pyod.models.combination import aom, moa, average, maximization
        from pyod.utils.data import generate_data
        from pyod.utils.utility import standardizer

        # train/test split with ground truth labels for evaluation
        X_train, X_test, y_train, y_test = generate_data(
            n_train=200, n_test=100, contamination=0.1)

        # standardize features before fitting
        X_train_norm, X_test_norm = standardizer(X_train, X_test)


2. Initialize 20 kNN outlier detectors with different k (10 to 200), and get the outlier scores.

    .. code-block:: python

        # initialize 20 base detectors for combination
        k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                    150, 160, 170, 180, 190, 200]
        n_clf = len(k_list)  # number of classifiers being trained

        train_scores = np.zeros([X_train.shape[0], n_clf])
        test_scores = np.zeros([X_test.shape[0], n_clf])

        for i in range(n_clf):
            k = k_list[i]
            clf = KNN(n_neighbors=k, method='largest')
            clf.fit(X_train_norm)

            train_scores[:, i] = clf.decision_scores_
            test_scores[:, i] = clf.decision_function(X_test_norm)

3. Then the output scores are standardized into zero average and unit std before combination.
   This step is crucial to adjust the detector outputs to the same scale.

    .. code-block:: python

        from pyod.utils.utility import standardizer

        # scores have to be normalized before combination
        train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

4. Four different combination algorithms are applied as described above:

    .. code-block:: python

        comb_by_average = average(test_scores_norm)
        comb_by_maximization = maximization(test_scores_norm)
        comb_by_aom = aom(test_scores_norm, 5) # 5 groups
        comb_by_moa = moa(test_scores_norm, 5) # 5 groups

5. Finally, all four combination methods are evaluated by ROC and Precision
   @ Rank n:

    .. code-block:: bash

        Combining 20 kNN detectors
        Combination by Average ROC:0.9194, precision @ rank n:0.4531
        Combination by Maximization ROC:0.9198, precision @ rank n:0.4688
        Combination by AOM ROC:0.9257, precision @ rank n:0.4844
        Combination by MOA ROC:0.9263, precision @ rank n:0.4688

