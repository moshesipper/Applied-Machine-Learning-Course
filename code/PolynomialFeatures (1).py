# copyright 2022 moshe sipper
# www.moshesipper.com

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_classification

X,y = make_circles(n_samples=300, shuffle=True, noise=0.1, factor=0.5)

plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.title('DATASET')
plt.show()

clf1 = LogisticRegression().fit(X, y)

plt.scatter(X[:,0], X[:,1], c=clf1.predict(X), cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.title('LOGISTIC PREDICTION')
plt.show()

xx = np.arange(-1, 1, 0.01)
yy = np.arange(-1, 1, 0.01)
XX,YY = np.meshgrid(xx,yy)
XY=np.array([XX.flatten(),YY.flatten()]).T

plt.scatter(XY[:,0], XY[:,1], c=clf1.predict(XY), cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.title('LOGISTIC DECISION BOUNDARY')
plt.show()

plt.scatter(XY[:,0], XY[:,1], c=clf1.predict_proba(XY)[:,1], cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.title('LOGISTIC DECISION BOUNDARY')
plt.show()

# [x0, x1]: the degree-2 polynomial features are [1, x0, x1, x0^2, x0*x1, x1^2].
X_poly = PolynomialFeatures(degree = 2).fit_transform(X)
clf2 = LogisticRegression().fit(X_poly, y)

plt.scatter(X[:,0], X[:,1], c=clf2.predict(X_poly), cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.title('LOGISTIC WITH POLYNOMIAL PREDICTION')
plt.show()

XY_poly = PolynomialFeatures(degree = 2).fit_transform(XY)
plt.scatter(XY[:,0], XY[:,1], c=clf2.predict(XY_poly), cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.title('LOGISTIC WITH POLYNOMIAL DECISION BOUNDARY')
plt.show()

XY_poly = PolynomialFeatures(degree = 2).fit_transform(XY)
plt.scatter(XY[:,0], XY[:,1], c=clf2.predict_proba(XY_poly)[:,1], cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.title('LOGISTIC WITH POLYNOMIAL DECISION BOUNDARY')
plt.show()


n_features=2
X,y = make_classification(n_samples=100, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0)
print(f'{n_features} features, poly:',PolynomialFeatures(degree = 2).fit_transform(X).shape[1])

n_features=20
X,y = make_classification(n_samples=100, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0)
print(f'{n_features} features, poly:',PolynomialFeatures(degree = 2).fit_transform(X).shape[1])

n_features=200
X,y = make_classification(n_samples=100, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0)
print(f'{n_features} features, poly:',PolynomialFeatures(degree = 2).fit_transform(X).shape[1])

n_features=2000
X,y = make_classification(n_samples=100, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0)
print(f'{n_features} features, poly:',PolynomialFeatures(degree = 2).fit_transform(X).shape[1])

'''
plt.rc('text', usetex=False)

X,y = make_circles(n_samples=300, shuffle=True, noise=0.1, factor=0.5)
clf1 = LogisticRegression().fit(X, y)
b = clf1.intercept_[0]
w1, w2 = clf1.coef_.T
c = -b/w2
m = -w1/w2
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0

xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
plt.scatter(*X[y==0].T, s=8, alpha=0.5)
plt.scatter(*X[y==1].T, s=8, alpha=0.5)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.show()

'''

'''
xi, yi = [], []
w=clf2.coef_.T
m1, m2 = int(100*xmin), int(100*xmax)
points = np.zeros((m2-m1+1,m2-m1+1))
for ix in range(m1,m2+1):
    for iy in range(m1,m2+1):
        x,y = ix/100, iy/100
        points[ix,iy] = clf2.predict([[1, x, y, x**2, x*y, y**2]])
        # points[ix,iy] = w[0] + w[1]*ix/100 + w[2]*iy/100  + w[3]*ix*ix/100/100  + w[4]*ix*iy/100/100  + w[5]*iy*iy/100/100 
        xi += [x]
        yi += [y]
plt.scatter(xi, yi, c=points, cmap='coolwarm',edgecolor='white', linewidth=0.3)
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.show()
'''

# w=clf.coef_.T

# Plot the data and the classification with the decision boundary.
# points = np.array([w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[0]*x[0] + w[4]*x[0]*x[1] + w[5]*x[1]*x[1] for x in X])
# plt.scatter(X[:,0], X[:,1], c=points, cmap='coolwarm',edgecolor='white', linewidth=0.3)
# plt.ylabel(r'$x_1$')
# plt.xlabel(r'$x_0$')
# plt.show()

# points = np.array([w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[0]*x[0] + w[4]*x[0]*x[1] + w[5]*x[1]*x[1] for x in zip(xx,yy)])
# plt.scatter(xx, yy, c=points, cmap='coolwarm',edgecolor='white', linewidth=0.3)
# plt.ylabel(r'$x_1$')
# plt.xlabel(r'$x_0$')
# plt.show()

# points = expit(points)

'''
X,y = make_circles(n_samples=300, shuffle=True, noise=0.1, factor=0.5)
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm',edgecolor='white', linewidth=0.3)
# plt.xlabel('$x_1$', fontsize=18)
# plt.ylabel('$y$', rotation=0, fontsize=18)
plt.show()


clf = LogisticRegression().fit(X,y)

line_bias = clf.intercept_
line_w = clf.coef_.T
points_y=[(line_w[0]*x+line_bias)/(-1*line_w[1]) for x in X]
plt.plot(X, points_y)
plt.scatter(X[:,0], X[:,1],c=y,cmap='coolwarm', edgecolor='white', linewidth=0.3)
plt.axis([5, 30, 0, 1])
plt.show()
'''


'''
X, y = make_circles(factor=0.7)
poly = PolynomialFeatures(degree = 2)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

X = X[:, :2]  # select 2 features

model = LogisticRegression()
clf = model.fit(X, y)


fig, ax = plt.subplots()
# title for the plots
# title = ('Decision surface of logistic regression ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_ylabel('y label here')
# ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
# ax.set_title(title)
ax.legend()
plt.show()
'''