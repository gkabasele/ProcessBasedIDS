import pdb
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

from welford import Welford
import utils

"""
Clustering of 1-D array
"""
class TimePattern(object):

    def __init__(self):
        self.values = []
        self.breakpoints = None
        self.clusters = None

    def update(self, value):
        self.values.append(value)

    def create_clusters(self):
        clusters, breakpoints = clustering_1D(self.values)
        if breakpoints is not None:
            self.breakpoints = breakpoints
            self.clusters = cluster_property(clusters)
        else:
            self.clusters = clusters
        self.values.clear()

    def __str__(self):
        return "(BP:{} C:{})".format(self.breakpoints, self.clusters)

    def __repr__(self):
        return self.__str__()

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

        if data[i-1] > data[i] and data[i+1] > data[i]:
            minima.append(i)

        if data[i-1] < data[i] and data[i+1] < data[i]:
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
        #breakpoints = [xs[i] for i in minimas]
        #clusters = [list() for i in range(len(minimas)+1)]
        breakpoints = [xs[i] for i in maximas]
        clusters = [list() for i in range(len(maximas))]
        for x in data:
            i = find_closest_bp(breakpoints, x)
            clusters[i].append(x)
            #for i, limit in enumerate(breakpoints):
            #    if x <= limit:
            #        clusters[i].append(x)
            #        break
            #    elif i == len(breakpoints)-1:
            #        clusters[i+1].append(x)
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


def main(filename):

    with open(filename, "r") as f:
        tmp = f.read().split(",")[:-1]
        data = [float(x) for x in tmp]
        pattern = TimePattern()
        for val in data:
            pattern.update(val)
        pattern.create_clusters()
        pdb.set_trace()

if __name__ == "__main__":
    main("./fit101dot027.txt")
