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
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
import jenkspy
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

def outlier_prob_score(data, model):
    if len(set(model.labels_)) > 2:
        soft_clusters = hdbscan.all_points_membership_vectors(model)
        return np.max(soft_clusters[-1])
    else:
        thresh_list, _ = run_soft_cluster(data, model, [-1])
        return thresh_list[-1]

def run_detection(normal, new_data, minPts, outlier_thresh, prob=True):
    point = np.reshape(new_data, (1, 2))
    data = np.append(normal, point, axis=0)
    model, labels = outlier_detected_recompute(data, minPts)
    if labels[-1] == -1:
        if prob:
            score = outlier_prob_score(data, model)
        else:
            score = inlier_score(data, labels, minPts)

        return score < outlier_thresh, score
    return False, None

#Code hdbscan for soft-clustering
#Part 1 Distance based membership
def exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(np.int)

def min_dist_to_exemplar(point, cluster_exemplars, data):
    dists = cdist([data[point]], data[cluster_exemplars.astype(np.int32)])
    return dists.min()

def dist_vector(point, exemplar_dict, data):
    result = {}
    for cluster in exemplar_dict:
        result[cluster] = min_dist_to_exemplar(point, exemplar_dict[cluster], data)
        return np.array(list(result.values()))

def dist_membership_vector(point, exemplar_dict, data, softmax=False):
    if softmax:
        result = np.exp(1./dist_vector(point, exemplar_dict, data))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = 1./dist_vector(point, exemplar_dict, data)
        result[~np.isfinite(result)] = np.finfo(np.double).max
    result /= result.sum()
    return result

#Part 2 outlier based membership
def max_lambda_val(cluster, tree):
    cluster_tree = tree[tree['child_size'] > 1]
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster)
    max_lambda = 0.0
    for leaf in leaves:
        max_lambda = max(max_lambda,
        tree['lambda_val'][tree['parent'] == leaf].max())
    return max_lambda

def points_in_cluster(cluster, tree):
    leaves = hdbscan.plots._recurse_leaf_dfs(tree, cluster)
    return leaves

def merge_height(point, cluster, tree, point_dict):
    ## multi point
    cluster_row = tree[tree["child"] == cluster]
    cluster_height = cluster_row["lambda_val"][0]
    
    if point in point_dict[cluster]:
        merge_row = tree[tree['child'] == float(point)][0]
        return merge_row['lambda_val']
    else:
        while point not in point_dict[cluster]:
            parent_row = tree[tree['child'] == cluster]
            cluster = parent_row['parent'].astype(np.float64)[0]
        for row in tree[tree['parent'] == cluster]:
            child_cluster = float(row['child'])
            if child_cluster == point:
                return row['lambda_val']
            if child_cluster in point_dict and point in point_dict[child_cluster]:
                return row['lambda_val']

def per_cluster_scores(point, cluster_ids, tree, max_lambda_dict, point_dict):
    result = {}
    point_row = tree[tree['child'] == point]
    point_cluster = float(point_row[0]['parent'])
    max_lambda = max_lambda_dict[point_cluster] + 1e-8 # avoid zero lambda vals in odd cases

    for c in cluster_ids:
        height = merge_height(point, c, tree, point_dict)
        result[c] = (max_lambda / (max_lambda - height))
    return result

def outlier_membership_vector(point, cluster_ids, tree,
                              max_lambda_dict, point_dict, softmax=True):
    if softmax:
        result = np.exp(np.array(list(per_cluster_scores(point,
                                                         cluster_ids,
                                                         tree,
                                                         max_lambda_dict,
                                                         point_dict
                                                        ).values())))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = np.array(list(per_cluster_scores(point,
                                                  cluster_ids,
                                                  tree,
                                                  max_lambda_dict,
                                                  point_dict
                                                 ).values()))
    result /= result.sum()
    return result

# Middle way
def combined_membership_vector(point, data, tree, exemplar_dict, cluster_ids,
                               max_lambda_dict, point_dict, softmax=False):
    raw_tree = tree._raw_tree
    dist_vec = dist_membership_vector(point, exemplar_dict, data, softmax)
    outl_vec = outlier_membership_vector(point, cluster_ids, raw_tree,
                                         max_lambda_dict, point_dict, softmax)
    result = dist_vec * outl_vec
    result /= result.sum()
    return result

def prob_in_some_cluster(point, tree, cluster_ids, point_dict, max_lambda_dict):
    heights = []
    for cluster in cluster_ids:
        heights.append(merge_height(point, cluster, tree._raw_tree, point_dict))
    height = max(heights)
    nearest_cluster = cluster_ids[np.argmax(heights)]
    max_lambda = max_lambda_dict[nearest_cluster]
    return height / max_lambda

def bfs_from_hierarchy(hierarchy, bfs_root):
    dim = hierarchy.shape[0]
    max_node = 2*dim
    num_points = max_node - dim + 1

    to_process = [bfs_root]
    result = []

    while to_process:
        result.extend(to_process)
        to_process = [x - num_points for x in
                      to_process if x >= num_points]
        if to_process:
            to_process = hierarchy[to_process,
                                   :2].flatten().astype(np.intp).tolist()
    return result

def hierarchy_example(data):
    hierarchy = linkage(data, "single")
    dn = dendrogram(hierarchy)

    root = 2* hierarchy.shape[0]
    num_points = root // 2 + 1
    next_label = num_points + 1

    node_list = bfs_from_hierarchy(hierarchy, root)
    relabel = np.empty(root + 1, dtype=np.intp)
    relabel[root] = num_points

    pdb.set_trace()

def run_soft_cluster(data, model, outliers_index):
    tree = model.condensed_tree_
    cluster_ids = tree._select_clusters()
    examplar_dict = {c:exemplars(c, tree) for c in cluster_ids}

    raw_tree = tree._raw_tree
    all_possible_clusters = np.arange(data.shape[0], raw_tree["parent"].max() + 1).astype(np.float64)
    max_lambda_dict = {c:max_lambda_val(c, raw_tree) for c in all_possible_clusters}
    point_dict = {c:set(points_in_cluster(c, raw_tree)) for c in all_possible_clusters}

    membership_vectors = np.empty((len(data), len(set(model.labels_)) - 1))
    for x in range(data.shape[0]):
        membership_vector = combined_membership_vector(x, data, tree, examplar_dict, cluster_ids,
                                                       max_lambda_dict, point_dict, False)
        membership_vector *= prob_in_some_cluster(x, tree, cluster_ids, point_dict, max_lambda_dict)

        membership_vectors[x] = membership_vector

    all_prob = [np.max(x) for x in membership_vectors]
    outliers_prob = [np.max(membership_vectors[i]) for i in outliers_index]

    return outliers_prob, all_prob
#DONE

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

    # No outliers have been detected so we cannot compute
    # a threshold
    if len(outliers_index) == 0:
        return None, None

    if prob:
        #implementation bug when there is only one cluster
        if len(set(clusters)) > 2:

            #outlier prob
            thresh_list, _ = compute_prob_outliers(model, outliers_index)
        else:
            thresh_list, _ = run_soft_cluster(data, model, outliers_index)

        param_space = [np.percentile(thresh_list, i) for i in range(25, 100, 25)]
    else:
        #local outlier factor
        thresh_list, _ = compute_lof_outliers(data, minPts, outliers_index)
        if len(thresh_list) > 3:
            # We consider two groups, the local outlier and the general outlier
            jnb = jenkspy.JenksNaturalBreaks(3)
            jnb.fit(thresh_list)
            return np.min(jnb.groups_[2]), thresh_list
        else:
            return -3, thresh_list

    return param_space[np.argmin(output_val)], thresh_list

def perform_detection(normal, attack, prob):
    pdb.set_trace()
    minPts = 4
    model = compute_hdbscan_model(normal, minPts)

    threshold, thresh_list = compute_threshold(normal, minPts, model, prob=prob)

    pdb.set_trace()

    try:
        plot_soft_clusters(normal, model, model.labels_,
                           outliers_index=np.where(model.labels_ == -1)[0],
                           text=False, info=thresh_list)
    except KeyError:
        plot_clusters(normal, model, model.labels_)

    perc_fake = list()
    perc_out = list()

    utils.plot_normal_attack(normal, attack)
    nbr_outliers = 0

    print("Threshold:{}".format(threshold))
    for new_data in enumerate(attack):
        new_val = new_data[1]
        is_outlier, score = run_detection(normal, new_val, minPts, threshold, prob)
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

        perform_detection(normal, attack, False)

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
