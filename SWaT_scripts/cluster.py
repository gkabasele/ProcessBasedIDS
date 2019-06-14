import os
import sys

from queue import Queue
import argparse
import pickle 
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, dest="input")
    parser.add_argument("--pv", type=str, dest="pv_name")
    args = parser.parse_args()

    data = pickle.load(open(args.input, "rb"))

    values = [x[args.pv_name] for x in data]
    val = np.array(values)
    val_2D = val.reshape(-1, 1)


    X = np.array([x for x in range(len(val_2D))])
    plt.plot(X, val_2D)
    plt.show()

    """
    km = KMeans(n_clusters=4)
    km.fit(val_2D)
    colors = ['r', 'g', 'b', 'c']
    centroids = km.cluster_centers_
    for n, y in enumerate(centroids):
        plt.plot(1, y, marker='x', color=colors[n], ms=10)
    plt.title('Kmeans cluster centroids')
    plt.show()

    z = km.predict(val_2D)

    n_clusters=4
    """


