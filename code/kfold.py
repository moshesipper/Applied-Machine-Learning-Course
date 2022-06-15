# Applied Machine Learning Course
# copyright 2022 moshe sipper
# www.moshesipper.com

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

X, y = load_iris(return_X_y=True)
# X, y = fetch_openml('credit-g', return_X_y=True, as_frame=False)
# kf = KFold(n_splits=3)
kf = KFold(n_splits=5,shuffle=True)
for train_index, test_index in kf.split(X):
    # print(test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape, X_test.shape)
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))
