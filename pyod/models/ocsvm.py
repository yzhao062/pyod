import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from scipy.special import erf

class OCSVM(OneClassSVM):

    def fit(self, X_train, y=None, sample_weight=None, **params):
        self.X_train = X_train
        super().fit(X=X_train, y=y, sample_weight=sample_weight, **params)
        return self

    def predict_proba(self, X_test, method='linear'):

        train_scores = self.decision_function(self.X_train) * -1
        test_scores = self.decision_function(X_test) * -1

        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            proba = scaler.transform(test_scores.reshape(-1, 1))
            return proba.clip(0, 1)
        else:

            mu = np.mean(train_scores)
            sigma = np.std(train_scores)

            # turn output into probability
            pre_erf_score = (test_scores - mu) / (sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            proba = erf_score.clip(0)

            # TODO: move to testing code
            assert (proba.min() >= 0)
            assert (proba.max() <= 1)

            return proba

    def predict_rank(self, X_test):

        train_scores = self.decision_function(self.X_train) * -1
        test_scores = self.decision_function(X_test) * -1
        ranks = np.zeros([X_test.shape[0], 1])

        for i in range(test_scores.shape[0]):
            train_scores_i = np.append(train_scores.reshape(-1, 1),
                                       test_scores[i])
            ranks[i] = rankdata(train_scores_i)[-1]

        # return normalized ranks
        ranks_norm = ranks / ranks.max()
        return ranks_norm