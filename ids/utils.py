import random
import pdb
import string
import yaml
import pickle
import collections
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from sklearn import preprocessing


# PORT
MODBUS_PORT = [5020, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5029,
               5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5038, 5039,
               5040]

# TAG
SRTAG_REDIRECT = 0
SRTAG_CLONE = 1

# TCP Flags

FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80

# Variable Type

DIS_COIL = "co"
DIS_INP = "di"
HOL_REG = "hr"
INP_REG = "ir"

DISCRETE = [DIS_COIL, DIS_INP]
CONTINOUS = [INP_REG, HOL_REG]
# Funcode write

WRITE_FUNCODE = [5, 6]

TS = "timestamp"
CAT = "normal/attack"

DIST = 0.01
DIFF = 0.05
DAY_IN_SEC = 86400
COOL_TIME = 11000

class RangeVal(object):

    __slots__= ["lower", "upper", "count", "norm"]

    def __init__(self, lower, upper, count, normalized=None):

        self.lower = lower
        self.upper = upper
        self.count = count
        self.norm = normalized

    def normalized(self, number):
        self.norm = self.count/number

    def __str__(self):
        return "{}-{}".format(self.lower, self.upper)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.lower == other.lower and self.upper == other.upper

    def hash(self):
        return hash(self.__str__())


class SetSWaT():
    def __init__(self):
        self.values = list()
        self.minval = None
        self.maxval = None

    def set_min_max(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def add(self, value):
        found = False
        for val in self.values:
            found = same_value(self.maxval, self.minval, val, value)
            if found:
                break

        if not found:
            self.values.append(value)

    def __iter__(self):
        return self.values.__iter__()

class CounterSWaT():

    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

        self.counter = collections.Counter()

    def count(self, values):

        for val in values:
            found = False
            for k, _ in self.counter.items():
                found = same_value(self.maxval, self.minval, k, val, thresh=0.01)
                if found:
                    self.counter.update([k])
                    break
            if not found:
                self.counter.update([val])

    def __iter__(self):
        return self.counter.__iter__()


class ProcessSWaTVar():

    def __init__(self, name, kind, min_val=None, max_val=None, 
                 digitizer=None, limit_values=None, ignore=False):
        self.name = name
        self.kind = kind
        self.value = None
        self.first = None
        self.nbr_transition = 0
        self.last_transition = None
        self.elapsed_time_transition = []
        self.ignore = ignore
        self.digitizer = digitizer

        self.is_periodic = kind in DISCRETE

        if min_val is None:
            self.min_val = 1
        else:
            self.min_val = min_val

        if max_val is None:
            self.max_val = 2
        else:
            self.max_val = max_val

        if limit_values is None:
            self.limit_values = []
        else:
            self.limit_values = limit_values

    def __hash__(self):
        return hash(self.name)

    def is_bool_var(self):
        return self.kind in [DIS_COIL, DIS_INP]

    def clear_time_value(self):
        self.first = None
        self.nbr_transition = 0
        self.elapsed_time_transition = []

    def normalized_dist(self, val1, val2):
        return (math.sqrt((val1-val2)**2)/math.sqrt((self.max_val - self.min_val)**2))

class ProcessVariable():

    def __init__(self, host, port, kind, addr, limit_values=None, gap=1,
                 size=None, name=None, first=None):
        self.host = host
        self.port = port
        self.kind = kind
        self.addr = addr
        self.name = name
        self.size = size
        self.gap = gap
        self.first = None
        self.nbr_transition = 0
        self.last_transition = None
        self.elapsed_time_transition = []
        self.current_ts = None
        self.value = None
        if limit_values:
            self.limit_values = limit_values
        else:
            self.limit_values = []

    @classmethod
    def funcode_to_kind(cls, funcode):
        if funcode in [1, 5, 15]:
            return DIS_COIL
        elif funcode == 2:
            return DIS_INP
        elif funcode in [3, 6, 10, 22, 23]:
            return HOL_REG
        elif funcode == 4:
            return INP_REG

    def key(self):
        return (self.host, self.port, self.kind, self.addr)

    def __eq__(self, other):
        return ((self.host, self.port, self.kind, self.addr) ==
                (other.host, other.port, other.kind, other.addr))

    def __hash__(self):
        return hash((self.host, self.port, self.kind, self.addr))

    def __str__(self):
        return "%s : (ip: %s, port: %s, type: %s, addr: %s)" % (
                                self.name, self.host, self.port, self.kind, self.addr) 
    def __repr__(self):
        return "%s : (ip: %s, port: %s, type: %s, addr: %s)" % (
                                self.name, self.host,self.port, self.kind, self.addr) 

    def is_bool_var(self):
        return self.kind in [DIS_COIL, DIS_INP]

    def clear_time_value(self):
        self.first = None
        self.nbr_transition = 0
        self.elapsed_time_transition = []

def setup(filename, pv_store):
    with open(filename) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for var_desc in desc['variables']:
            var = var_desc['variable']
            if var['type'] == DIS_COIL or var['type'] == DIS_INP:
                limit_values = [1, 2]
                min_val = 1
                max_val = 2
            else:
                limit_values = var['values']
                min_val = var['min']
                max_val = var['max']

            pv = ProcessSWaTVar(var['name'], var['type'],
                                limit_values=limit_values,
                                min_val=min_val,
                                max_val=max_val)
            pv_store[pv.name] = pv

def randomName(stringLength=4):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def is_number(s):
    """ Returns Truse if string s is a number """
    return s.replace('.','',1).isdigit()

def same_value(max_val, min_val, val1, val2, thresh=DIST, noisy=True):
    if noisy:
        return normalized_dist(max_val, min_val, val1, val2) <= thresh
    else:
        return val1 == val2

def normalized_dist(max_val, min_val, val1, val2):
    return (math.sqrt((val1-val2)**2)/math.sqrt((max_val - min_val)**2))

def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid')/window

def show_kde(data, name=None):
    xs, y_data = compute_kde(data) 
    n, bins, patches = plt.hist(data, 150, density=True)
    if name is not None:
        plt.title(name)
    plt.plot(xs, y_data)
    plt.show()

def compute_kde(data):
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 150)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    y_data = density(xs)
    return xs, y_data

def plot_1d(data):
    val = 0.
    plt.plot(data, np.zeros_like(data) + val, 'x')
    plt.show()

def read_state_file(name):
    with open(name, "rb") as filename:
        data = pickle.load(filename)
    return data

# Export time to evaluate time pattern IDS
def export_time_pattern(inname, outfile):
    data = read_state_file(inname)
    with open(outfile, "w") as fname:
        for state in data:
            ts = state['timestamp']
            fname.write("{}: 0\n".format(ts))


def get_all_values_pv(data, pvname, limit=None):
    if limit is None:
        return np.array([x[pvname] for x in data])
    else:
        return np.array([x[pvname] for x in data[:limit]])

def get_colors_vectorizer():
    colors = ["royalblue", "maroon", "forestgreen", "mediumorchid",
              "tan", "deeppink", "olive", "goldenrod", "lightcyan",
              "navy"]

    return np.vectorize(lambda x: colors[x % len(colors)])

def plot_clusters(X, cluster_mapping):
    vectorizer = get_colors_vectorizer()
    plt.scatter(X[:, 0], X[:, 1], c=vectorizer(cluster_mapping))
    plt.show()

def plot_datapoint(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

def plot_normal_attack(normal, attack):
    plt.subplot(211)
    plt.scatter(normal[:, 0], normal[:, 1])
    plt.subplot(212)
    plt.scatter(attack[:, 0], attack[:, 1])
    plt.show()


def plot_clusters_with_outlier(X, cluster_mapping, new_data, mapping_new_data):
    vectorizer = get_colors_vectorizer()
    plt.scatter(X[:, 0], X[:, 1], c=vectorizer(cluster_mapping), alpha=0.5)
    plt.scatter(new_data[:, 0], new_data[:, 1],
                c=vectorizer(mapping_new_data), marker="^")
    plt.show()

def save_plot_clusters(X, cluster_mapping, filename):
    vectorizer = get_colors_vectorizer()
    plt.scatter(X[:, 0], X[:, 1], c=vectorizer(cluster_mapping))
    plt.savefig(filename)

def normalized(X, x_min=0, x_max=1):
    nom = (X-X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom/denom

def standardize(X):
    return preprocessing.scale(X)
