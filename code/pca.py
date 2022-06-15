# Applied Machine Learning Course
# copyright 2022 moshe sipper
# www.moshesipper.com

# Load the iris data
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Fit a PCA
from sklearn.decomposition import PCA
n_components = 2
pca = PCA(n_components=n_components, whiten=True)
pca.fit(X)

# Project the data in 2D
X_pca = pca.transform(X)

# Visualize the data
target_ids = range(len(iris.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
for i, c, label in zip(target_ids, 'rgbcmykw', iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
               c=c, label=label)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


explained_variance = pca.explained_variance_ratio_
with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(n_components), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()

