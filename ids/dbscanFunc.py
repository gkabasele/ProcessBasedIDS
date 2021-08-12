import argparse
import os
import itertools
import math
from timeit import default_timer as timer
from collections.abc import Iterable
import numpy as np
import pdb
from scipy.spatial.distance import euclidean 
from scipy.spatial.distance import cdist 
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
import jenkspy
import seaborn as sns
import hdbscan
from matplotlib import pyplot as plt

import utils
from DBCV import DBCV

OUTLIERS = 0
NORMAL = 1


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
    minpts = None

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
                minPts = i

    if max_model is not None:
        return max_model, minPts

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
    nbr_outliers = sum([1 for x in outlier_lof if x < threshold])
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
            hdbscan_model, _ = compute_hdbscan_model(matrix, minPts)
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
    model_hdb, _ = compute_hdbscan_model(X, minPts)
    clusters = model_hdb.labels_

    utils.plot_clusters(X, clusters)

    print("DBSCAN FPR: {}".format(compute_outlier_fraction(model)))
    print("HDBSCAN FPR: {}".format(compute_outlier_fraction(model_hdb)))

    pdb.set_trace()

def outlier_detected_predict(model, new_point):
    labels, _ = hdbscan.approximate_predict(model, new_point)
    return labels

def outlier_detected_recompute(data, minPts):
    model, _ = compute_hdbscan_model(data, minPts)
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
    model, _ = compute_hdbscan_model(X, 3)

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

    if text or info is None:
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

def three_sigma_rule(data, new_point):
    mu = np.mean(data)
    std = np.std(data)
    return abs(new_point-mu) > 3 * std

def onedim_iqr(data, new_point):

    d = np.sort(data)
    q1 = np.percentile(d, 25, interpolation="midpoint")
    q2 = np.percentile(d, 50, interpolation="midpoint")
    q3 = np.percentile(d, 75, interpolation="midpoint")
    iqr = q3 - q1
    low_lim = q1 - 1.5 * iqr
    up_lim = q3 + 1.5 * iqr
    return new_point > up_lim or new_point < low_lim

def run_detection_forest(model, data, new_data, threshold, stillness, debug):
    if model is None or threshold is None:
        return new_data in data

    return model.decision_function(new_data)[0] < threshold

def run_detection_svm(model, data, new_data, stillness, debug):
    if model is None:
        return new_data in data
    return model.predict(new_data)[0] < 0


def inlier_score(attack, minPts, debug=False):
    clf = LocalOutlierFactor(n_neighbors=minPts, contamination="auto")
    clf.fit_predict(attack)
    if debug:
        pdb.set_trace()
    return clf.negative_outlier_factor_[-1]

def run_detection(normal, new_data, minPts, outlier_thresh,
                  stillness, prob=False, debug=False):
    #if stillness:
    #    #return onedim_iqr(normal[:, 1], new_data[0][1]), None
    #    return three_sigma_rule(normal[:, 1], new_data[0][1]), None

    point = np.reshape(new_data, (1, 2))
    data = np.append(normal, point, axis=0)

    score = inlier_score(data, minPts, debug)
    if debug:
        pdb.set_trace()


    return score < outlier_thresh, score

    # This might not be needed
    #model, labels = outlier_detected_recompute(data, minPts)

    #if labels[-1] == -1:
    #    score = inlier_score(data, minPts)
    #    return score < outlier_thresh, score

    #return False, None


def compute_lof_outliers(data, minPts, outliers_index):
    clf = LocalOutlierFactor(n_neighbors=minPts, contamination="auto")
    clf.fit_predict(data)
    return clf.negative_outlier_factor_

def compute_prob_outliers(model, outliers_index):

    soft_clusters = hdbscan.all_points_membership_vectors(model)

    all_prob = [np.max(x) for x in soft_clusters]
    outliers_prob = [np.max(soft_clusters[i]) for i in outliers_index]
    return outliers_prob, all_prob

def compute_threshold(data, minPts, model, prob=False, display=False):
    clusters = model.labels_
    outliers_index = np.where(clusters == -1)[0]

    # No outliers have been detected so we cannot compute
    # a threshold, therefore is fixed to -3
    #if len(outliers_index) == 0:
    #    print("No outliers")
    #    return -3, None

    #local outlier factor
    lof = compute_lof_outliers(data, minPts, outliers_index)

    if len(lof) > 2:
        ## We consider two groups, the local outlier and the general outlier
        jnb = jenkspy.JenksNaturalBreaks(2)
        jnb.fit(lof)

        if display:
            print("Thresh List:{}".format(jnb.groups_))
        try:
            min_val = np.min(jnb.groups_[NORMAL])
        except ValueError:
            min_val = np.min(jnb.groups_[OUTLIERS])
        return min_val, lof
        #thresh = np.quantile(lof, .02)
        #thresh = np.min(lof)
        #return thresh, lof
    else:
        return -3, lof


def perform_detection(normal, attack, prob):
    minPts = 10
    if len(normal) < minPts:
        raise ValueError("The dataset is to small")
    model, _ = compute_hdbscan_model(normal, minPts)

    threshold, thresh_list = compute_threshold(normal, minPts, model, prob=prob)
    if threshold is None:
        print("No Threshold could be found assigning  one")
        threshold = -3

    """
    try:
        plot_soft_clusters(normal, model, model.labels_,
                           outliers_index=np.where(model.labels_ == -1)[0],
                           text=False, info=thresh_list)
    except KeyError:
        plot_clusters(normal, model, model.labels_)
    """

    perc_fake = list()
    perc_out = list()

    utils.plot_normal_attack(normal, attack)
    nbr_outliers = 0

    print("Threshold:{}".format(threshold))
    for new_data in enumerate(attack):
        new_val = new_data[1]
        is_outlier, score = run_detection(normal, new_val, minPts, threshold, prob)
        if is_outlier:
            #print("Outlier: {} ,score: {}".format(np.reshape(new_val, (1, 2)), score))
            nbr_outliers += 1

    if thresh_list is not None:
        nbr_fake_outlier = compute_outlier(threshold, thresh_list)
        nbr_fake_outlier_ratio = nbr_fake_outlier/len(normal)
        print("Nbr outlier norm: {} ({}/{})".format(nbr_fake_outlier_ratio, nbr_fake_outlier, len(normal)))
        perc_fake.append(nbr_fake_outlier_ratio)

    outlier_ratio = nbr_outliers/len(attack)
    print("Nbr outlier atk: {} ({}/{})".format(outlier_ratio, nbr_outliers, len(attack)))
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

def count_outlier(predictions, data):
    nbr_outlier = len(np.where(predictions == -1)[0])

    print("Nbr Anomalies: {}/{} ({})".format(nbr_outlier, len(data),
                                             nbr_outlier/len(data)))
    return nbr_outlier

def one_class_svm(data_normal, data_attack):
    clf = OneClassSVM(gamma="auto").fit(data_normal)

    predictions = clf.predict(data_normal)
    print("Prediction: \n")
    print(predictions)

    count_outlier(predictions, data_normal)
    
    predictions = clf.predict(data_attack)

    count_outlier(predictions, data_attack)
    print("Attack: \n")
    print(predictions)

    new_point = [[0, 2000]]
    prediction = clf.predict(new_point)
    print("New data point: \n")
    print(prediction)

def isolation_forest(data_normal, data_attack, max_features=1, n_estimators=10,
                     max_samples="auto", contamination="auto"):

    model = IsolationForest(max_features=max_features,
                            n_estimators=n_estimators, max_samples=max_samples,
                            contamination=contamination, warm_start=True, bootstrap=True,
                            random_state=1)
    print("Data normal")
    model.fit(data_normal)

    print("Isolation normal anomaly result: \n")
    predictions = model.predict(data_normal)
    nbr_outlier = len(np.where(predictions == -1)[0])
    print("Nbr Anomalies: {}/{} ({})".format(nbr_outlier, len(data_normal),
                                             nbr_outlier/len(data_normal)))


    score = model.decision_function(data_normal)
    outliers_score = [score[i] for i in np.where(score < 0)]
    most_outlier_score = min(score)
    print("outliers score: {}".format(outliers_score))
    print("Max outlier score: {}".format(most_outlier_score)) 

    print("Data Attack")
    print("Isolation attack anomaly result: \n")
    predictions = model.predict(data_attack)
    nbr_outlier = len(np.where(predictions == -1)[0])
    print("Nbr Anomalies: {}/{} ({})".format(nbr_outlier, len(data_attack),
                                             nbr_outlier/len(data_attack)))

    nbr_less_thresh = len(np.where(model.decision_function(data_attack) <= most_outlier_score)[0])
    print("Nbr anomalies thresh: {}/{} ({})".format(nbr_less_thresh, len(data_attack),
                                                    nbr_less_thresh/len(data_attack)))
    new_point = [[0, 2000]]
    print("New stupid data point result")
    pred = model.predict(new_point)
    print(pred)
    print("New stupid  data point score")
    score = model.decision_function(new_point)
    print(score)

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

        perform_detection(normal, attack, False)

    #one_class_svm(normal, attack)

    start = timer()
    print("Parameters default")
    #isolation_forest(normal, attack)
    end = timer()
    print("Isolation 2D: {}".format(end - start))

    isolation_forest(normal, attack, max_features=1, n_estimators=10, max_samples=0.6,
                     contamination=0.05)

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
