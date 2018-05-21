'''
Example of using ABOD for outlier detection
'''
import os
import pathlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from pyod.data.load_data import generate_data
from pyod.models.abod import ABOD
from pyod.utils.utility import precision_n_scores

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200
    n_test = 100

    X_train, y_train, c_train, X_test, y_test, c_test = generate_data(
        n_train=n_train, n_test=n_test, contamination=contamination)

    # train a ABOD detector (default version)
    clf = ABOD(contamination=contamination, fast_method=False)
    clf.fit(X_train)

    # get the prediction on the training data
    y_train_pred = clf.y_pred
    y_train_score = clf.decision_scores * -1

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)
    y_test_score = clf.decision_function(X_test) * -1

    print('Train ROC:{roc}, precision@n:{prn}'.format(
        roc=roc_auc_score(y_train, y_train_score),
        prn=precision_n_scores(y_train, y_train_score)))

    print('Test ROC:{roc}, precision@n:{prn}'.format(
        roc=roc_auc_score(y_test, y_test_score),
        prn=precision_n_scores(y_test, y_test_score)))

    #######################################################################
    # Visualizations
    # initialize the log directory if it does not exist
    pathlib.Path('example_figs').mkdir(parents=True, exist_ok=True)

    # plot the results
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(221)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=c_train)
    plt.title('Train ground truth')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='normal',
                              markerfacecolor='b', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='outlier',
                              markerfacecolor='r', markersize=8)]

    plt.legend(handles=legend_elements, loc=4)

    ax = fig.add_subplot(222)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=c_test)
    plt.title('Test ground truth')
    plt.legend(handles=legend_elements, loc=4)

    ax = fig.add_subplot(223)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
    plt.title('Train prediction by ABOD')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='normal',
                              markerfacecolor='0', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='outlier',
                              markerfacecolor='yellow', markersize=8)]
    plt.legend(handles=legend_elements, loc=4)

    ax = fig.add_subplot(224)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred)
    plt.title('Test prediction by ABOD')
    plt.legend(handles=legend_elements, loc=4)

    plt.savefig(os.path.join('example_figs', 'abod.png'), dpi=300)

    plt.show()
