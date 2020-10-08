import argparse
import os
import itertools
from timeit import default_timer as timer
from collections.abc import Iterable
import numpy as np
import pdb
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns
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
    # min_sample = K of the KNN
    min_samples = [i for i in range(1, min(len(data), n))]
    max_dbcv_score = None
    max_model = None

    for i in min_samples:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=n, min_samples=i,
                                    prediction_data=True,
                                    gen_min_span_tree=True,
                                    allow_single_cluster=True,
                                    approx_min_span_tree=True,
                                    metric="euclidean")
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

def compute_dbscan_model(distances, data, n, pre_eps=None):

    if pre_eps is not None:
        dbscan = DBSCAN(eps=pre_eps, min_samples=n)
        model = dbscan.fit(data)
        return model

    epsilon = [np.quantile(distances, x*0.1) for x in range(1, 9)]

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

def compute_outlier(threshold, outlier_lof):
    nbr_outliers = sum([1 for x in outlier_lof if x <  threshold])
    return nbr_outliers

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

def outlier_detected_recompute(data, minPts):
    model = compute_hdbscan_model(data, minPts)
    clusters = model.labels_
    return model, clusters

def outlier_detected_recompute_dbscan(model, data, new_data, minPts,
                                      distances):
    points = np.append(data, new_data, axis=0)
    new_model = compute_dbscan_model(distances, data, minPts,
                                     pre_eps=model.eps)
    clusters = new_model.labels_
    return clusters


def outlier_test():

    X, y = make_blobs(n_samples=150, centers=3,
                      cluster_std=0.5, random_state=0)
    minPts = 3
    model = compute_hdbscan_model(X, 3)

    clusters = model.labels_
    utils.plot_clusters(X, clusters)

    pdb.set_trace()

def plot_clusters(data, model, clusters, outliers_index=None, text=True, prob=None):
    color_palette = sns.color_palette('Paired', len(clusters))
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusters]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, model.probabilities_)]

    plt.scatter(data[:, 0], data[:, 1], s=50, linewidth=0, c=cluster_member_colors, alpha=0.75)

    if text:
        add_text_to_plot(data, outliers_index)
    else:
        add_info_to_plot(data, outliers_index, prob)

    plt.show()

def plot_soft_clusters(data, model, clusters, outliers_index=None, text=True, info=None):
    soft_clusters = hdbscan.all_points_membership_vectors(model)

    color_palette = sns.color_palette('Paired', len(clusters))
    cluster_member_colors = [color_palette[np.argmax(x)]
                             for x in soft_clusters]
    plt.scatter(data[:, 0], data[:, 1], s=50, linewidth=0, c=cluster_member_colors, alpha=0.75)

    if text:
        add_text_to_plot(data, outliers_index)
    else:
        if isinstance(info[0], Iterable):
            round_text = [np.round(v[1], 5) for v in info]
        else:
            round_text = [np.round(v, 5) for v in info]

        add_info_to_plot(data, outliers_index, round_text)

    plt.show()

def add_text_to_plot(data, outliers_index):
    for i in range(data.shape[0]):
        if outliers_index is not None:
            if i in outliers_index:
                plt.text(data[i, 0], data[i, 1], str(i))
        else:
            plt.text(data[i, 0], data[i, 1], str(i))

def add_info_to_plot(data, outliers_index, info):
    for out_index, i in enumerate(outliers_index):
        plt.text(data[i, 0], data[i, 1], str(info[out_index]))

def get_outlier_dist_per_cluster(map_point_cluster, data_lof, clusters, outliers_index):
    cluster_outlier_dist = {x: list() for x in clusters if x != -1}
    cluster_outlier_lof_dist = {x: list() for x in clusters if x != -1}


    for i in outliers_index:
        cluster, prob = map_point_cluster[i]
        cluster_outlier_dist[cluster].append(prob)
        cluster_outlier_lof_dist[cluster].append(data_lof[i])

    for k in cluster_outlier_dist:
        if len(cluster_outlier_dist[k]) > 0:
            cluster_outlier_dist[k] = (len(cluster_outlier_dist[k]),
                                       np.mean(cluster_outlier_dist[k]),
                                       np.var(cluster_outlier_dist[k]))

    return cluster_outlier_dist, cluster_outlier_lof_dist

def inlier_score(attack, clusters, minPts):
    clf = LocalOutlierFactor(n_neighbors=minPts-1, contamination='auto')
    clf.fit_predict(attack)
    return clf.negative_outlier_factor_[-1]

def outlier_prob_score(model, minPts):
    soft_clusters = hdbscan.all_points_membership_vectors(model)
    return np.max(soft_clusters[-1])

def run_detection(normal, new_data, minPts, outlier_thresh, prob=True):
    point = np.reshape(new_data, (1, 2))
    data = np.append(normal, point, axis=0)
    model, labels = outlier_detected_recompute(data, minPts)
    if labels[-1] == -1:
        if prob:
            score = outlier_prob_score(model, minPts)
        else:
            score = inlier_score(data, labels, minPts)

        return score < outlier_thresh, score
    return False, None

def compute_lof_outliers(data, minPts, outliers_index): 
    clf = LocalOutlierFactor(n_neighbors=minPts-1, contamination="auto")
    clf.fit_predict(data)
    all_lof = [clf.negative_outlier_factor_[i] for i in outliers_index]
    return all_lof, clf

def compute_prob_outliers(model, outliers_index):

    soft_clusters = hdbscan.all_points_membership_vectors(model)
    all_prob = [np.max(x) for x in soft_clusters]
    outliers_prob = [np.max(soft_clusters[i]) for i in outliers_index]
    return outliers_prob, all_prob

def compute_threshold(data, minPts, model, prob=True):
    clusters = model.labels_
    outliers_index = np.where(clusters == -1)[0]

    if prob:
        #outlier prob
        thresh_list, _ = compute_prob_outliers(model, outliers_index)
        param_space = [np.percentile(thresh_list, i) for i in range(25, 100, 25)]

    else:
        #local outlier factor
        tmp, _ = compute_lof_outliers(data, minPts, outliers_index)
        thresh_list = [x for x in tmp if x > -2]
        param_space = [np.percentile(thresh_list, i) for i in range(10, 100, 10)]

    output_val = [compute_outlier(x, thresh_list)/len(data) for x in param_space]

    return param_space[np.argmin(output_val)], thresh_list

def perform_detection(normal, attack, prob):
    minPts = 4
    model = compute_hdbscan_model(normal, minPts)

    threshold, thresh_list = compute_threshold(normal, minPts, model, prob=prob)

    pdb.set_trace()

    plot_soft_clusters(normal, model, model.labels_,
                       outliers_index=np.where(model.labels_ == -1)[0],
                       text=False, info=thresh_list)

    perc_fake = list()
    perc_out = list()

    #utils.plot_normal_attack(normal, attack)
    nbr_outliers = 0

    print("Threshold:{}".format(threshold))
    for new_data in enumerate(attack):
        new_val = new_data[1]
        is_outlier, score = run_detection(normal, new_val, minPts, threshold)
        if is_outlier:
            print("Outlier: {} ,score: {}".format(np.reshape(new_val, (1, 2)), score))
            nbr_outliers += 1

    nbr_fake_outlier = compute_outlier(threshold, thresh_list)
    nbr_fake_outlier_ratio = nbr_fake_outlier/len(normal)

    outlier_ratio = nbr_outliers/len(attack)
    print("Nbr outlier norm: {} ({}/{})".format(nbr_fake_outlier_ratio, nbr_fake_outlier, len(normal)))
    print("Nbr outlier atk: {} ({}/{})".format(outlier_ratio, nbr_outliers, len(attack)))
    perc_fake.append(nbr_fake_outlier_ratio)
    perc_out.append(outlier_ratio)

def plot_thresh_impact(perc, fake_ratio, outlier_ratio):

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("percentile")
    ax1.set_ylabel("fake", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(perc, fake_ratio, color=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("outlier", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.plot(perc, outlier_ratio, color=color)

    fig.tight_layout()
    plt.show()

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

        perform_detection(normal, attack, True)

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
