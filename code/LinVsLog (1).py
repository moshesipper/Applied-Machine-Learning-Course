import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_redundant=0)
# X, y = make_classification(n_samples=10000, n_features=5000, n_informative=5000, n_redundant=0)
X, y = make_classification(n_samples=1000, n_features=5000, n_informative=5000, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LinearRegression().fit(X_train, y_train)
pred = clf.predict(X_test)
pred = np.where(pred < 0.5, 0, 1)
acc = accuracy_score(y_test, pred)
print('LinearRegression test acc', acc)

clf = LogisticRegression().fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print('LogisticRegression test acc', acc)