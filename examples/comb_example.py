import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.load_data import load_cardio, load_letter
from models.knn import Knn
from models.combination import aom
from utility.utility import get_precn

if __name__ == "__main__":

    # number of estimators
    n_clf = 20
    ite = 20

    # load data
    X, y = load_cardio()

    # roc mean
    roc_mean = []
    roc_max = []
    roc_aom = []
    roc_moa = []

    for t in range(ite):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.4)

        # standardizing data for processing
        scaler = StandardScaler().fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)

        k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                  150, 160, 170, 180, 190, 200]

        train_scores = np.zeros([X_train.shape[0], n_clf])
        test_scores = np.zeros([X_test.shape[0], n_clf])

        for i in range(n_clf):
            k = k_list[i]

            clf = Knn(n_neighbors=k, method='largest')
            clf.fit(X_train_norm)

            train_scores[:, i] = clf.decision_scores.ravel()
            test_scores[:, i] = clf.decision_function(X_test_norm).ravel()
            # print(k, roc_auc_score(y_test, test_scores[:, i]))

        # scores have to be normalized before combination
        scaler = StandardScaler().fit(train_scores)
        train_scores_norm = scaler.transform(train_scores)
        test_scores_norm = scaler.transform(test_scores)

        mean_result = np.mean(test_scores_norm, axis=1)
        roc_mean.append(roc_auc_score(y_test, mean_result))
        print('ite', t, 'mean', roc_auc_score(y_test, mean_result))

        max_result = np.max(test_scores_norm, axis=1)
        roc_max.append(roc_auc_score(y_test, max_result))
        print('ite', t, 'max', roc_auc_score(y_test, max_result))

        aom_result = aom(test_scores_norm, 5, 20)
        roc_aom.append(roc_auc_score(y_test, aom_result))
        print('ite', t, 'aom', roc_auc_score(y_test, aom_result))

        moa_result = aom(test_scores_norm, 5, 20)
        roc_moa.append(roc_auc_score(y_test, moa_result))
        print('ite', t, 'moa', roc_auc_score(y_test, moa_result))

        print()

    print('summary')
    print('mean roc:', np.mean(roc_mean))
    print('max roc:', np.mean(roc_max))
    print('aom roc:', np.mean(roc_aom))
    print('moa roc:', np.mean(roc_moa))
