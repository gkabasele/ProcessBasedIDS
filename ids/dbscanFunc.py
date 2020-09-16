import numpy as np
import pdb
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib import pyplot as plt

import utils
from DBCV import DBCV



def compute_knn_distance(data, n):

    neigh = NearestNeighbors(n_neighbors=n)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    return distances

def remove_outliers(data, labels):
    outliers_index = np.where(labels == -1)

    cola = np.delete(data[:, 0], outliers_index)
    colb = np.delete(data[:, 1], outliers_index)

    new_data = np.array(list(zip(cola, colb)))

    return new_data, np.delete(labels, outliers_index)

def compute_dbscan_model(distances, data, n):

    epsilon = [np.quantile(distances, x*0.1) for x in range(1, 11)]

    max_dbcv_score = None
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
                dbcv_score = DBCV(data, labels, dist_function=euclidean)

                if np.isnan(dbcv_score):
                    continue

                if max_dbcv_score is None or dbcv_score > max_dbcv_score:
                    max_dbcv_score = dbcv_score
                    max_model = model

    if max_model is not None:
        return max_model

    else:
        print("passing here")
        e = max(1/10**6, np.quantile(distances, 0.5))
        dbscan = DBSCAN(eps=e, min_samples=n)
        return dbscan.fit(data)


def main():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.4, random_state=0)

    minPts = 3

    distances = compute_knn_distance(X, minPts)
    model = compute_dbscan_model(distances, X, minPts)
    clusters = model.labels_

    utils.plot_clusters(X, clusters)

    print("Found Epsilon: {}".format(model.eps))
    plt.plot(distances)
    plt.show()

    pdb.set_trace()

if __name__ == "__main__":
    main()
