import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

from welford import Welford

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
        self.breakpoints = breakpoints
        self.clusters = cluster_property(clusters)
        self.values.clear()

def find_minima_local(data):
    minima = []
    for i in range(1, len(data)-1):
        if data[i-1] >= data[i] and data[i+1] >= data[i]:
            minima.append(i)

    return minima

def clustering_1D(data):
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 150)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    y_data = density(xs)
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
    main("./time.txt")
