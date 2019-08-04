import pdb
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

def find_minima_local(data):
    minima = []
    for i in range(1, len(data)-1):
        if data[i-1] >= data[i] and data[i+1] >= data[i]:
            minima.append(i)

    return minima

def clustering_1D(data):
    try:
        xs, y_data = utils.compute_kde(data)
        minimas = find_minima_local(y_data)
        breakpoints = [xs[i] for i in minimas]
        clusters = [list() for i in range(len(minimas)+1)]
        for x in data:
            for i, limit in enumerate(breakpoints):
                if x <= limit:
                    clusters[i].append(x)
                    break
                elif i == len(breakpoints)-1:
                    clusters[i+1].append(x)
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
