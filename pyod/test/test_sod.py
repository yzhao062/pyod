from pyod.models.sod import SOD
from pyod.utils import generate_data


'''
TO-DO by Yue Zhao
'''
X_train, y_train, X_test, y_test = generate_data(n_train=100, n_test=0, n_features=10,
                                                 contamination=0.1, random_state=0)
#print(X_train)
sod = SOD(contamination=0.1, n_neighbors=15, ref_set=10, alpha=0.8)
sod.fit(X_train)
print(sod.decision_scores_)
print(sod.threshold_)
print(sod.labels_)