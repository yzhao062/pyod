import os
import pandas as pd
from pyador.pyador import Pyador
from pyador import local as const
import numpy as np
from models.knn import Knn

if __name__ == "__main__":
    # dev_file = os.path.join(const.DEV_DIR, const.DEV_FILE)
    # X = pd.read_csv(dev_file)
    # print("training data is %s " % dev_file)
    # print("data shape is %s" % (X.shape,))
    #
    # # initialize the program with fraud percentage
    # clf = Pyador(frac=0.05)
    # clf._debug()
    #
    # y_pred, X_train, Y_train = clf.fit(X)
    # print(X_train.shape, Y_train.shape)
    samples = [[-1, 0], [0., 0.], [1., 1], [2., 5.], [3, 1]]

    clf = Knn()
    clf.fit(samples)
    print(clf.sample_scores(np.asarray([[2, 3], [6, 8]])))
    clf.evaluate(np.asarray([[2, 3], [6, 8]]), np.asarray([[0], [1]]))
