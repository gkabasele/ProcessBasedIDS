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

    def __init__(self, value):
        self.values = [value]
        self.wel = Welford(value)

    def update(self, value):
        self.values.append(value)
        self.wel(value)

def find_minima_local(data):
    minima = []
    maxima = []
    for i in range(1, len(data)-1):
        if data[i-1] >= data[i] and data[i+1] >= data[i]:
            minima.append(i)

        if data[i-1] <= data[i] and data[i+1] <= data[i]:
            maxima.append(i)
    return minima, maxima

def clustering_1D(data):
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 150)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    y_data = density(xs)
    plt.hist(data, 100, density=True)
    plt.plot(xs, y_data)
    #plt.show()
    minimas, maximas = find_minima_local(y_data)
    print("Min:{}, Max:{}".format(minimas, maximas))
    clusters = [list() for i in range(len(minimas)+1)]
    print([xs[i] for i in minimas])

    for x in data:
        for i, index in enumerate(minimas):
            if x <= xs[index]:
                clusters[i].append(x)
                break
            elif i == len(minimas)-1:
                clusters[i+1].append(x)
    for cluster in clusters:
        print(cluster)
    """
    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(data)

    # score samples returns the log of the probability density
    logprob = kde.score_samples(data)
    plt.hist(data, bins=100)
    plt.plot(data, np.full_like(data, 
    """

def main(filename):

    with open(filename, "r") as f:
        tmp = f.read().split(",")[:-1]
        data = [float(x) for x in tmp]

        clustering_1D(data)
main("./time.txt")
