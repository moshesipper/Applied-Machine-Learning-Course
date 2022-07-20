# copyright 2022 moshe sipper
# www.moshesipper.com

import time
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer

Algorithms = [Ridge, Lasso, LogisticRegression, LinearSVC, SGDClassifier, KNeighborsClassifier, DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, MLPClassifier]

X, y = load_breast_cancer(return_X_y=True)

times = {}
for alg in Algorithms:
    clf = alg()
    tic = time.time()
    clf.fit(X, y)
    toc = time.time()
    times[alg.__name__] = toc - tic

srt = dict(sorted(times.items(), key=lambda item: item[1]))
for k, v in srt.items():
    print(k, v)
