# copyright 2022 moshe sipper
# www.moshesipper.com

import time
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

Algorithms = [LogisticRegression, LinearSVC, SGDClassifier, KNeighborsClassifier, DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, MLPClassifier]


# compare time
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


# compare acc
clfs = {}
for alg in Algorithms:
    clfs[alg.__name__] = {'alg': alg(), 'acc': 0}

X, y = load_breast_cancer(return_X_y=True)

n_splits=5
kf = KFold(n_splits=n_splits)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    for alg in Algorithms:
        clfs[alg.__name__]['alg'].fit(X_train, y_train)
        clfs[alg.__name__]['acc'] += accuracy_score(y_test, clfs[alg.__name__]['alg'].predict(X_test))

for alg in Algorithms:
    clfs[alg.__name__]['acc'] /= n_splits

srt = dict(sorted(clfs.items(), key=lambda x: x[1]['acc'], reverse=True))
for k, v in srt.items():
    print(k, f'{v["acc"]:.3f}')
