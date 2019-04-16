# -*- coding: utf-8 -*-
"""Example of using and ``visualizing generate_data_clusters`` function
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: MIT
from __future__ import division
from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.utils.data import get_outliers_inliers

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))


def visualize(X_train, y_train, show_figure=True, save_figure=False):
    """Utility function for visualizing the results in examples.
    Internal use only.

    Parameters
    ----------
    X_train : numpy array of shape (n_samples, n_features)
        The training samples.

    y_train : list or array of shape (n_samples,)
        The ground truth of training samples.

    show_figure : bool, optional (default=True)
        If set to True, show the figure.

    save_figure : bool, optional (default=False)
        If set to True, save the figure to the local.

    """

    def _plot(X_inliers, X_outliers, inlier_color='blue', outlier_color='orange'):
        """Internal method to add subplot of inliers and outliers.

        Parameters
        ----------
        X_inliers : numpy array of shape (n_samples, n_features)
            Outliers.

        X_outliers : numpy array of shape (n_samples, n_features)
            Inliers.

        sub_plot_title : str
            Subplot title.

        inlier_color : str, optional (default='blue')
            The color of inliers.

        outlier_color : str, optional (default='orange')
            The color of outliers.

        """
        plt.axis("equal")
        plt.scatter(X_inliers[:, 0], X_inliers[:, 1], label='inliers',
                    color=inlier_color, s=40)
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                    label='outliers', color=outlier_color, s=50, marker='^')
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc='best', prop={'size': 10})

    assert len(X_train) <= 5
    in_colors = ['blue', 'green', 'purple', 'brown', 'black']
    out_colors = ['red', 'orange', 'grey', 'violet', 'pink']
    plt.figure(figsize=(13, 10))
    plt.suptitle("Demo of Generating Data in Clusters", fontsize=15)
    for i, cluster in enumerate(X_train):
        X_train_outliers, X_train_inliers = get_outliers_inliers(cluster, y_train[i])
        _plot(X_train_inliers, X_train_outliers,
              inlier_color=in_colors[i],
              outlier_color=out_colors[i])

    if save_figure:
        plt.savefig()

    if show_figure:
        plt.show()


if __name__ == "__main__":
    contamination = 0.05  # percentage of outliers
    n_samples = 500  # number of sample points
    test_size = None  # ratio of testing points - None means no test split is wanted

    # Generate sample data in clusters
    x, y = generate_data_clusters(n_samples=n_samples,
                                  test_size=test_size,
                                  n_clusters=3,
                                  n_features=2,
                                  contamination=contamination,
                                  size='different',
                                  density='different',
                                  dist=0.2,
                                  random_state=42,
                                  return_in_clusters=True)

    # visualize the results
    visualize(x, y, show_figure=True, save_figure=False)
