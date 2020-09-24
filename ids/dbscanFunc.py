import argparse
import os
from timeit import default_timer as timer
import numpy as np
import pdb
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
import hdbscan
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

def compute_hdbscan_model(data, n):

    # number of neighbour to be a core point
    min_samples = [i for i in range(2, min(len(data), 10))]
    max_dbcv_score = None
    max_model = None

    for i in min_samples:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=n, min_samples=i,
                                    prediction_data=True,
                                    gen_min_span_tree=True,
                                    allow_single_cluster=True)
        model = clusterer.fit(data)
        labels = clusterer.labels_
        labels_unique = set(labels)
        clusters_nbr = len(labels_unique) - (1 if -1 in labels else 0)

        if clusters_nbr > 0:

            score = clusterer.relative_validity_
            if max_dbcv_score is None or score > max_dbcv_score:
                max_dbcv_score = score
                max_model = model

    if max_model is not None:
        return max_model

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

def compute_outlier_fraction(model):
    nbr_outliers = (model.labels_ == -1).sum()
    return nbr_outliers/len(model.labels_)

def compare_clustering_from_dir(directory):
    dbscan_outlier = list()
    hdbscan_outlier = list()
    dbscan_time = list()
    hdbscan_time = list()
    minPts = 3

    for f_dir in os.listdir(directory):
        filename = os.path.join(directory, f_dir)
        print("Starting file: {}".format(filename))
        with open(filename, "rb") as f:
            matrix = np.load(f)
            print("Nbr of datapoints: {}".format(len(matrix)))

            # DBSCAN
            start = timer()
            distances = compute_knn_distance(matrix, minPts)
            dbscan_model = compute_dbscan_model(distances, matrix, minPts)
            end = timer()
            dbscan_time.append(end - start)

            # HDBSCAN
            start = timer()
            hdbscan_model = compute_hdbscan_model(matrix, minPts)
            end = timer()
            hdbscan_time.append(end - start)

            dbscan_outlier.append(compute_outlier_fraction(dbscan_model))
            hdbscan_outlier.append(compute_outlier_fraction(hdbscan_model))

    print("DBSCAN FPR, m:{} v:{}".format(np.mean(dbscan_outlier),
                                         np.var(dbscan_outlier)))
    print("DBSCAN Exec time, m:{}, v:{}".format(np.mean(dbscan_time),
                                                np.var(dbscan_time)))

    print("HDBSCAN FPR, m:{} v:{}".format(np.mean(hdbscan_outlier),
                                          np.var(hdbscan_outlier)))
    print("HDBSCAN Exec time, m:{}, v:{}".format(np.mean(hdbscan_time),
                                                 np.var(hdbscan_time)))

def compare_clustering_from_file(matrix=None):
    if matrix is None:
        X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.7, random_state=0)
    else:
        X = matrix

    minPts = 3

    print("DBSCAN")

    distances = compute_knn_distance(X, minPts)
    model = compute_dbscan_model(distances, X, minPts)
    clusters = model.labels_

    utils.plot_clusters(X, clusters)

    print("HDBSCAN")
    model_hdb = compute_hdbscan_model(X, minPts)
    clusters = model_hdb.labels_

    utils.plot_clusters(X, clusters)

    print("DBSCAN FPR: {}".format(compute_outlier_fraction(model)))
    print("HDBSCAN FPR: {}".format(compute_outlier_fraction(model_hdb)))

    pdb.set_trace()

def outlier_detected_predict(model, new_point):
    labels, _ = hdbscan.approximate_predict(model, new_point)
    return labels

def outlier_detected_recompute(model, data, new_data, minPts):
    points = np.append(data, new_data, axis=0)
    model = compute_hdbscan_model(points, minPts)
    clusters = model.labels_
    return clusters

def outlier_test():

    X, y = make_blobs(n_samples=150, centers=3,
                      cluster_std=0.5, random_state=0)
    minPts = 3
    model = compute_hdbscan_model(X, 3)

    clusters = model.labels_
    utils.plot_clusters(X, clusters)

    pdb.set_trace()

def perform_detection(normal, attack):
    minPts = 3
    model = compute_hdbscan_model(normal, minPts)
    clusters = model.labels_

    nbr_outliers = 0

    utils.plot_clusters(normal, clusters)

    utils.plot_normal_attack(normal, attack)

    for i, new_data in enumerate(attack):
        point = np.reshape(new_data, (1, 2))
        label = outlier_detected_recompute(model, normal,
                                           point, minPts)
        if label[-1] == -1:
            print("Outlier on position: {} for value {}".format(i, point))
            nbr_outliers += 1
            #utils.plot_clusters_with_outlier(normal, clusters, point, label)

    print("Nbr outlier norm: {}".format(compute_outlier_fraction(model)))
    print("Nbr outlier atk: {}".format(nbr_outliers/len(attack)))

def main(filename, is_dir, normal_trace, attack_trace):
    normal = None
    attack = None
    if filename is not None:
        if not is_dir:
            with open(filename, "rb") as f:
                matrix = np.load(f)
                compare_clustering_from_file(matrix)
        else:
            compare_clustering_from_dir(filename)
    else:
        with open(normal_trace, "rb") as f:
            normal = np.load(f)

        with open(attack_trace, "rb") as f:
            attack = np.load(f)

        perform_detection(normal, attack)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="filename",
                        help="matrix file of values")
    parser.add_argument("--normal", action="store", dest="normal",
                        help="training matrix of the normal trace")
    parser.add_argument("--attack", action="store", dest="attack",
                        help="validation matrix of the attack trace")
    parser.add_argument("--dir", action="store_true", dest="is_dir",
                        help="indicate if the filename point to a directory")
    args = parser.parse_args()
    main(args.filename, args.is_dir, args.normal, args.attack)
