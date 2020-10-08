import pdb
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
import hdbscan
from welford import Welford
import dbscanFunc
import utils

"""
Clustering of 1-D array
"""
class TimePattern(object):

    def __init__(self, minPts=4, same_crit_val=True):
        #time transition
        self.values = list()
        # updates means for transition
        self.steps = list()
        self.breakpoints = None
        self.clusters = None
        self.same_crit_val = same_crit_val
        self.model = None
        self.min_pts = minPts
        self.data = None

        self.threshold = None

    def update(self, value):
        self.values.append(value)

    def add_update_step(self, val):
        self.steps.append(val)

    def create_clusters(self):
        clusters, breakpoints = clustering_1D(self.values)
        if breakpoints is not None:
            self.breakpoints = breakpoints
            self.clusters = cluster_property(clusters)
        else:
            self.clusters = clusters

    def get_cluster(self, data):
        arr = [x.mean for x in self.clusters]
        i = find_closest_bp(arr, data)
        cluster = self.clusters[i]
        return cluster

    def export_data_matrix(self, filename):
        with open(filename, "wb") as f:
            if self.data is None:
                self.data = self.get_matrix_from_data(self.steps, self.values)
            np.save(f, self.data)

    def compute_clusters(self, strategy="hdbscan", name=None, row=None, col=None):
        print("Name:{}, Row:{}, Col:{}".format(name, row, col))
        self.data = self.get_matrix_from_data(self.steps, self.values)

        if len(self.data) >= self.min_pts:
            if strategy:
                self.model = dbscanFunc.compute_hdbscan_model(self.data, self.min_pts)
                self.threshold = dbscanFunc.compute_threshold(self.data, self.min_pts, self.model)
        else:
            self.model = self.data

    def get_matrix_from_data(self, steps, values):
        data = np.array(list(zip(steps, values)))
        return utils.standardize(data)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()

    def is_outlier_hdbscan(self, time_elapsed, update_step):
        if len(self.data) > self.min_pts:
            is_outlier, _ = dbscanFunc.run_detection(self.data,
                                                     np.array([[update_step, time_elapsed]]),
                                                     self.min_pts,
                                                     self.threshold)
            return is_outlier
        else:
            return [update_step, time_elapsed] in self.model


def find_extreme_local(data):
    minima = []
    maxima = []
    for i in range(len(data)):
        if i == 0:
            if data[i+1] > data[i]:
                minima.append(i)
            elif data[i+1] < data[i]:
                maxima.append(i)
            continue

        if i == len(data)-1:
            if data[i-1] > data[i]:
                minima.append(i)
            elif data[i-1] < data[i]:
                maxima.append(i)
            continue

        if data[i-1] > data[i] and data[i+1] >= data[i]:
            minima.append(i)

        if data[i-1] < data[i] and data[i+1] <= data[i]:
            maxima.append(i)

    return minima, maxima

def get_closest(arr, pos1, pos2, target):
    val1 = arr[pos1]
    val2 = arr[pos2]
    if target - val1 >= val2 - target:
        return pos2
    else:
        return pos1

def find_closest_bp(arr, target):
    n = len(arr)

    if target <= arr[0]:
        return 0
    if target >= arr[n-1]:
        return n-1

    i = 0
    j = n
    mid = 0

    while i < j:
        mid = math.floor((i + j)/2)

        if arr[mid] == target:
            return mid

        if target < arr[mid]:
            # Case for non integer value
            if mid > 0 and target > arr[mid-1]:
                return get_closest(arr, mid-1, mid, target)
            j = mid
        else:
            if mid < n-1 and target < arr[mid + 1]:
                return get_closest(arr, mid, mid + 1, target)
            i = mid + 1
    return mid

def clustering_1D(data):
    # Cluster are created by looking for minimal local in the KDE
    # Then each data are associated to a cluster to compute the mean
    # and variance of the cluster
    try:
        xs, y_data = utils.compute_kde(data)
        minimas, maximas = find_extreme_local(y_data)
        # No maximas has been found, can occur if constant value
        if len(maximas) == 0:
            wel = Welford()
            wel(data)
            clusters = [wel]
            breakpoints = None
        else:
            breakpoints = [xs[i] for i in maximas]
            clusters = [list() for i in range(len(maximas))]
            if len(breakpoints) == 0:
                pdb.set_trace()
            for x in data:
                i = find_closest_bp(breakpoints, x)
                clusters[i].append(x)
    except np.linalg.LinAlgError:
        wel = Welford()
        wel(data)
        clusters = [wel]
        breakpoints = None

    except ValueError as err:
        if len(data) == 1:
            wel = Welford()
            wel(data)
            clusters = [wel]
            breakpoints = None
        else:
            raise(err)

    return clusters, breakpoints

def cluster_property(clusters):
    properties = []
    for cluster in clusters:
        wel = Welford()
        wel(cluster)
        properties.append(wel)
    return properties


def test_cluster_kde():
    a = np.array([10, 11, 9, 23, 21, 11, 45, 20, 11, 12]).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    s = np.linspace(0, 50)
    e = kde.score_samples(s.reshape(-1, 1))


    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    print("Minima", s[mi])
    print("Maxima", s[ma])

    print(a[a<mi[0]], a[(a>=mi[0]) * (a<= mi[1])], a[a >= mi[1]])

    pattern = TimePattern()
    data = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
    for val in data:
        pattern.update(val)
    pattern.create_clusters()
    print(pattern)

    plt.plot(s, e)
    plt.plot(s[:mi[0]+1], e[:mi[0]+1], 'r',
             s[mi[0]:mi[1]+1], e[mi[0]:mi[1]+1], 'g',
             s[mi[1]:], e[mi[1]:], 'b',
             s[ma], e[ma], 'go',
             s[mi], e[mi], 'ro')
    plt.show()

def main(filename):

    with open(filename, "r") as f:
        data = [float(x) for x in f]
        pattern = TimePattern()
        for val in data:
            pattern.update(val)
        pattern.create_clusters()
        pdb.set_trace()
        print(pattern)

if __name__ == "__main__":
    main("./time_pattern_test/fit502dot1245.txt")
