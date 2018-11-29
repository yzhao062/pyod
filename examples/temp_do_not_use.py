import pandas as pd
from pyod.models.sos import SOS
iris = pd.read_csv("http://bit.ly/iris-csv")
X = iris.drop("Name", axis=1).values
detector = SOS()
detector.fit(X)
iris["score"] = detector.decision_scores_

print(iris.sort_values("score", ascending=False).head(10))