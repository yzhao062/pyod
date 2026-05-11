# -*- coding: utf-8 -*-
"""Example of saving and loading PyOD models.

The recommended flow uses :func:`pyod.utils.persistence.save` and
:func:`pyod.utils.persistence.load`, which wrap ``joblib`` with a
versioned envelope that records the dependency versions in effect at
save time. ``load`` reads that envelope and warns when sklearn, joblib,
numpy, or scipy drift between save and load environments; it also
falls through to ``compat_load`` automatically when sklearn's Tree
node dtype evolves and an older artifact would otherwise fail to load.

The raw ``joblib.dump`` / ``joblib.load`` flow still works and is
included as a secondary alternative for users who want full control
over the file format.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.lof import LOF
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.utils.persistence import save, load


if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # train LOF detector
    clf_name = 'LOF'
    clf = LOF()
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # ---- Recommended: save / load with a versioned envelope ----
    artifact_path = 'clf.pyod.joblib'
    save(clf, artifact_path, metadata={'dataset': 'demo', 'note': 'LOF baseline'})

    # The matching load() reads the envelope and warns on dependency
    # drift; pass strict=True for version-pinned production deployments.
    clf = load(artifact_path)

    # To inspect the envelope without separately re-reading the file:
    clf, env = load(artifact_path, return_metadata=True)
    print(
        f"Loaded {env['model_class']} "
        f"(pyod={env['pyod_version']}, sklearn={env['sklearn_version']}, "
        f"saved_at={env['saved_at']})")

    # ---- Alternative: raw joblib.dump / joblib.load ----
    # The legacy raw-joblib path still works. It does not record
    # dependency versions, and cross-sklearn-version compatibility is
    # the user's responsibility. Prefer save() / load() above for new
    # code.
    #
    #     from joblib import dump as _dump, load as _load
    #     _dump(clf, 'clf.joblib')
    #     clf = _load('clf.joblib')

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # visualize the results
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
