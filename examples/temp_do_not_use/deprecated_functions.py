import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.utils import check_X_y


def visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True,
              save_figure=False):  # pragma: no cover
    """
    Utility function for visualizing the results in examples
    Internal use only

    :param clf_name: The name of the detector
    :type clf_name: str

    :param X_train: The training samples
    :param X_train: numpy array of shape (n_samples, n_features)

    :param y_train: The ground truth of training samples
    :type y_train: list or array of shape (n_samples,)

    :param X_test: The test samples
    :type X_test: numpy array of shape (n_samples, n_features)

    :param y_test: The ground truth of test samples
    :type y_test: list or array of shape (n_samples,)

    :param y_train_pred: The predicted outlier scores on the training samples
    :type y_train_pred: numpy array of shape (n_samples, n_features)

    :param y_test_pred: The predicted outlier scores on the test samples
    :type y_test_pred: numpy array of shape (n_samples, n_features)

    :param show_figure: If set to True, show the figure
    :type show_figure: bool, optional (default=True)

    :param save_figure: If set to True, save the figure to the local
    :type save_figure: bool, optional (default=False)
    """

    if X_train.shape[1] != 2 or X_test.shape[1] != 2:
        raise ValueError("Input data has to be 2-d for visualization. The "
                         "input data has {shape}.".format(shape=X_train.shape))

    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)
    c_train = get_color_codes(y_train)
    c_test = get_color_codes(y_test)

    fig = plt.figure(figsize=(12, 10))
    plt.suptitle("Demo of {clf_name}".format(clf_name=clf_name))

    fig.add_subplot(221)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=c_train)
    plt.title('Train ground truth')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='inlier',
                              markerfacecolor='b', markersize=8),
                       Line2D([0], [0], marker='^', color='w', label='outlier',
                              markerfacecolor='r', markersize=8)]

    plt.legend(handles=legend_elements, loc=4)

    fig.add_subplot(222)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=c_test)
    plt.title('Test ground truth')
    plt.legend(handles=legend_elements, loc=4)

    fig.add_subplot(223)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
    plt.title('Train prediction by {clf_name}'.format(clf_name=clf_name))
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='inlier',
                              markerfacecolor='0', markersize=8),
                       Line2D([0], [0], marker='^', color='w', label='outlier',
                              markerfacecolor='yellow', markersize=8)]
    plt.legend(handles=legend_elements, loc=4)

    fig.add_subplot(224)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred)
    plt.title('Test prediction by {clf_name}'.format(clf_name=clf_name))
    plt.legend(handles=legend_elements, loc=4)

    if save_figure:
        plt.savefig('{clf_name}.png'.format(clf_name=clf_name), dpi=300)
    if show_figure:
        plt.show()
    return


def get_color_codes(y):
    """Internal function to generate color codes for inliers and outliers.
    Inliers (0): blue; Outlier (1): red.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    Returns
    -------
    c : numpy array of shape (n_samples,)
        Color codes.

    """
    y = column_or_1d(y)

    # inliers are assigned blue
    c = np.full([len(y)], 'b', dtype=str)
    outliers_ind = np.where(y == 1)

    # outlier are assigned red
    c[outliers_ind] = 'r'

    return c
