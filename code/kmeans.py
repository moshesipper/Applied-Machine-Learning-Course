# Applied Machine Learning Course
# copyright 2022 moshe sipper
# www.moshesipper.com

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.cluster import KMeans
clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)

from matplotlib import pyplot as plt
plt.plot(range(1, 11), clust)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('var')
plt.show()

