import os
import numpy as np
from scipy.io import loadmat


def generate_data(n=1000, contamination=0.1, n_test=500):
    n_outliers = int(n * contamination)
    n_inliers = int(n - n_outliers)

    n_outliers_test = int(n_test * contamination)
    n_inliers_test = int(n_test - n_outliers_test)

    offset = 2

    # generate inliers
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    X = np.r_[X1, X2]

    # generate outliers
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # generate target
    y = np.zeros([X.shape[0], 1])
    c = np.full([X.shape[0]], 'b', dtype=str)
    y[n_inliers:, ] = 1
    c[n_inliers:, ] = 'r'

    # generate test data
    X1_test = 0.3 * np.random.randn(n_inliers_test // 2, 2) - offset
    X2_test = 0.3 * np.random.randn(n_inliers_test // 2, 2) + offset
    X_test = np.r_[X1_test, X2_test]

    # generate outliers
    X_test = np.r_[
        X_test, np.random.uniform(low=-8, high=8, size=(n_outliers_test, 2))]
    y_test = np.zeros([X_test.shape[0], 1])

    c_test = np.full([X_test.shape[0]], 'b', dtype=str)

    y_test[n_inliers_test:] = 1
    c_test[n_inliers_test:] = 'r'

    return X, y, c, X_test, y_test, c_test


def load_cardio():
    mat = loadmat(os.path.join('resources', 'cardio.mat'))
    X = mat['X']
    y = mat['y'].ravel()

    return X, y


def load_letter():
    mat = loadmat(os.path.join('resources', 'letters.mat'))
    X = mat['X']
    y = mat['y'].ravel()

    return X, y
