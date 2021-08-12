import pdb
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
import hdbscan
from welford import Welford
import dbscanFunc
import utils

MAX_THRESHOLD = 100000

"""
Clustering of 1-D array
"""
class TimePattern(object):

    def __init__(self, minPts=5, bool_var=False, same_crit_val=True):
        #time transition
        self.values = list()
        # updates means for transition
        self.steps = list()
        self.breakpoints = None
        self.clusters = None
        self.same_crit_val = same_crit_val
        self.model = None
        # Stillness time of boolean value are too similar (there are some duplicate)
        # which result to very big LOF, so we need to consider more points
        self.min_pts = minPts
        self.data = None
        self.max_time = None

        self.bool_var = bool_var

        self.threshold = None

        self.forest = None

        self.svm = None

        
    def update(self, value):
        self.values.append(value)

    def add_update_step(self, val):
        self.steps.append(val)

    def export_data_matrix(self, dirname, name, row, col):
        filename = dirname + "{}_{}_{}.bin".format(name, row, col)
        with open(filename, "wb") as f:
            if self.data is None:
                self.data = self.get_matrix_from_data(self.steps, self.values)
            np.save(f, self.data)

    def compute_clusters(self, strategy="hdbscan", name=None, row=None, col=None):
        print("Name:{}, Row:{}, Col:{}".format(name, row, col))
        self.max_time = np.max(self.values)
        self.data = self.get_matrix_from_data(self.steps, self.values)
        #self.export_data_matrix("./test_swat_transition_dataset_attack/", name, row, col)

        if len(self.data) >= self.min_pts:
            self.model, self.min_pts = dbscanFunc.compute_hdbscan_model(self.data, self.min_pts)
            self.threshold, _ = dbscanFunc.compute_threshold(self.data, self.min_pts, self.model, False)
            if self.threshold > MAX_THRESHOLD:
                min_pts = self.min_pts
                while self.threshold > MAX_THRESHOLD and min_pts <= len(self.data):
                    min_pts += 1
                    self.threshold, _ = dbscanFunc.compute_threshold(self.data, min_pts, self.model, False)
                if min_pts > len(self.data):
                    self.min_pts = len(self.data)
                    self.model = self.data
                else:
                    self.min_pts = min_pts

        else:
            self.model = self.data

    def train_svm(self, name=None, row=None, col=None):
        print("Name:{}, Row:{}, Col:{}".format(name, row, col))
        self.max_time = np.max(self.values)
        self.data = self.get_matrix_from_data(self.steps, self.values)
        if len(self.data) > 0:
            self.svm = OneClassSVM(gamma="auto").fit(self.data)

    def train_forest(self, name=None, row=None, col=None):
        print("Name:{}, Row:{}, Col:{}".format(name, row, col))
        self.max_time = np.max(self.values)
        self.data = self.get_matrix_from_data(self.steps, self.values)
        if len(self.data) > 0:
            #isolation Forest approach
            max_features = 1
            n_estimators = 15
            max_samples = int(0.6*len(self.data))
            if max_samples == 0:
                max_samples = len(self.data)
            contamination = 0.05
            self.forest = IsolationForest(max_features=max_features,
                                          n_estimators=n_estimators, max_samples=max_samples,
                                          contamination=contamination, warm_start=True,
                                          bootstrap=True, random_state=1)

            self.forest.fit(self.data)
            score = self.forest.decision_function(self.data)
            outlier_score = [score[i] for i in np.where(score < 0)[0]]
            if len(outlier_score) > 0:
                self.threshold = max(outlier_score)

    def get_matrix_from_data(self, steps, values):
        data = np.array(list(zip(steps, values)))
        return data

    def __str__(self):
        if len(self.data) >= self.min_pts:
            clusters = len(set(self.model.labels_)) -1 if -1 in self.model.labels_ else 0
            nbr_outlier = len(np.where(clusters == -1)[0])
            return str("#Clusters:{}, #Outliers:{}, #Points:{}".format(clusters, nbr_outlier, self.data))
        else:
            clusters = len(self.data)
            return str("#Clusters:{}".format(clusters))

    def __repr__(self):
        return self.__str__()

    def is_outlier_hdbscan(self, time_elapsed, update_step, stillness, debug=False):
        if self.threshold is not None:
            if len(self.data) > self.min_pts:
                if debug:
                    pdb.set_trace()
                is_outlier, score = dbscanFunc.run_detection(self.data,
                                                             np.array([[update_step, time_elapsed]]),
                                                             self.min_pts,
                                                             self.threshold, stillness, False, debug)
                return is_outlier, score
            else:
                return [update_step, time_elapsed] in self.model, None

        return True, None

    def is_outlier_forest(self, time_elapsed, update_step, stillness, debug=False):
        is_outlier = dbscanFunc.run_detection_forest(self.forest, self.data,
                                                     np.array([[update_step, time_elapsed]]),
                                                     self.threshold, stillness, debug)

        return is_outlier, None

    def is_outlier_svm(self, time_elapsed, updata_step, stillness, debug=False):
        is_outlier = dbscanFunc.run_detection_svm(self.svm, self.data,
                                                  np.array([[updata_step, time_elapsed]]),
                                                  stillness, debug)
        return is_outlier, None

    def has_cluster(self):
        return len(self.data) > self.min_pts


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
