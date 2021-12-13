"""
K-Means Clustering

K is the number of centroids which represent the center of a cluster
Don't include labels for data points
Include the features/variables of each data
Requires computer to determine what are the qualifications  of a feature

1. Given K centroids, can randomly assign the position of K centroids
2. Repeat
    3. Assign each point to its nearest centroid
    4. Compute the new centroid of each cluster
5. Until the centroid position do not change

Disadvantage
Very slow, requires a lot of computation
Points * Centroids * Iterations * Features

Advantage
Still faster than some other clustering algorithms
"""

import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)  # .data is all the features and all features are scaled to be between -1 and 1 to save
# computation
y = digits.target

k = len(np.unique(y))  # dynamically sizes k if dataset is changed
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


clf = KMeans(n_clusters=k, init="random", n_init=10)
# n_clusters: How many things to be classified
# K-means++ for centroids to be initialized equal distance from each other, random for random initial location
# n_init: Amount of different seeds with result taken from the best seeds
# max_iter: Maximum amount of iterations in a single run
bench_k_means(clf, "1", data)

