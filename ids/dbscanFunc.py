import numpy as np
import pdb
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib import pyplot as plt

import utils



def compute_knn_distance(data, n):

    neigh = NearestNeighbors(n_neighbors=n)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    return distances


def compute_dbscan_model(distances, data, n):

    epsilon = [np.quantile(distances, x*0.1) for x in range(1,11)]

    max_silhouette = -2
    max_model = None

    for e in epsilon:
        if e != 0:
            dbscan = DBSCAN(eps=e, min_samples=n)
            model = dbscan.fit(data)
            labels = model.labels_
            labels_unique = set(labels)
            clusters_nbr = len(labels_unique) - (1 if -1 in labels else 0)
            if len(labels_unique) == 1:
                continue

            if clusters_nbr > 0:
                sil = metrics.silhouette_score(data, labels)

                if sil > max_silhouette:
                    max_silhouette = sil
                    max_model = model

    if max_model is not None:
        return max_model
    else:
        e = max(1/10**6, np.quantile(distances, 0.5))
        dbscan = DBSCAN(eps=e, min_samples=n)
        return dbscan.fit(data)


def main():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)

    minPts = 3

    distances = compute_knn_distance(X, minPts)
    model = compute_dbscan_model(distances, X, minPts)
    clusters = model.labels_

    utils.plot_clusters(X, clusters)

    print("Found Epsilon: {}".format(model.eps))
    plt.plot(distances)
    plt.show()

if __name__ == "__main__":
    main()
