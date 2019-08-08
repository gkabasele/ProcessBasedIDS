import pdb
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import networkx as nx
import igraph as ig
from networkx.algorithms import community
import utils
import string
from scipy import stats
from pvStore import PVStore
from limitVal import RangeVal

NBR_RANGE = 10
class Edge(object):

    __slots__ = ['neigh', 'score']

    def __init__(self, neigh, score):
        self.neigh = neigh
        self.score = score

    def __hash__(self):
        return hash(self.neigh)

    def __eq__(self, other):
        return self.neigh == other.neigh

class Graph(object):

    def __init__(self):
        self.graph = {}

    def add_edge(self, node, neigh):
        if node not in self.graph:
            self.graph[node] = set([Edge(neigh,0)])
        else:
            if neigh in self.graph[node]:
                pass
                #edge = self.gra
            else:
                self.graph[node].add(neigh)

    def show_edges(self):
        for node in self.graph:
            for neigh in self.graph[node]:
                print("(", node, ", ", neigh, ")")

    def find_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        for node in self.graph[start]:
            if node not in path:
                newPath = self.find_path(node, end, path)
                if newPath:
                    return newPath
                return None

class Symbol(object):
    i = 0

    def __init__(self, val=None):
        if val is None:
            self.val = str(Symbol.i)
            Symbol.i += 1
        else:
            self.val = val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return self.__str__()

class SymbolEntry(object):

    def __init__(self, symbol, time=None):
        self.symbol = symbol
        self.start_time = time
        self.end_time = time

    def same_symbol(self, other):
        return self.symbol.val == other.val

    def val(self):
        return self.symbol.val

    def __hash__(self):
        return hash(self.symbol.val)

    def __eq__(self, other):
        return self.symbol.val == other.symbol.val

    def __str__(self):
        return str(self.symbol.val)

    def __repr__(self):
        return self.__str__()


class Digitizer(object):

    def __init__(self, len_dataset, min_val, max_val, graph):
        self.mapping = {}
        self.min_val = min_val
        self.max_val = max_val
        self.ranges = self.compute_ranges(len_dataset)
        self.graph = graph

    def compute_ranges(self, len_dataset):
        ranges = []
        #nbr_ranges = math.ceil(math.sqrt(len_dataset))
        nbr_ranges = NBR_RANGE
        ranges_width = (self.max_val - self.min_val)/nbr_ranges

        for i in range(nbr_ranges):
            lower = self.min_val + i * ranges_width
            upper = self.min_val + (i+1)*ranges_width
            r = RangeVal(lower, upper, 0)
            ranges.append(r)

        return ranges

    def nodes(self):
        return self.graph.nodes

    def edges(self):
        return self.graph.edges

    def get_range(self, x):
        for i, rangeval in enumerate(self.ranges):
            if x >= rangeval.lower and x <= rangeval.upper:
                return i, rangeval

    def create_graph(self, data):

        for i in range(len(data) - 1):
            x, curr_range = self.get_range(data[i])
            y, next_range = self.get_range(data[i+1])

            if curr_range != next_range:
                if self.graph.has_edge(x, y):
                    self.graph[x][y]['score'] += 1
                else:
                    self.graph.add_edge(x, y, score=1)

    def digitize(self, data):
        res = []

        for val in data:
            i, rangeval = self.get_range(val)
            res.append(i)
        return res

def symbol(val):
    return SymbolEntry(Symbol(val))

def test_digitize():
    digitizer = Digitizer([3, 6, 9], 0, 1)
    print(digitizer.mapping)
    input_l = [2, 5, 8, 1, 11, 3, 7, 2, 9, 13, 6, 11, 9]
    res = digitizer.digitize(input_l)
    print(res)
    assert [x.symbol.val for x in res] == ['a', 'c', 'e', 'a', 'g', 'b', 'e', 'a', 'f', 'g', 'd', 'g', 'f']

    prefix = [symbol('e'), symbol('a')]

    freq = search_symbols_periodicity(res, prefix) 

    print(freq)

def _search_symbols_periodicity(subsequence, prefix):
    res = True
    for a, v in zip(subsequence, prefix):
        res = a == v
        if not res:
            break
    return res

def search_symbols_periodicity(sequence, prefix):
    periods = []
    i = 0
    while i <= len(sequence) - len(prefix):
        if sequence[i] == prefix[0]:
            res = _search_symbols_periodicity(sequence[i:i+len(prefix)], prefix)
            if res:
                periods.append(i)
                i += len(prefix)
            else:
                i += 1
        else:
            i += 1
    freq = [] 
    for i in range(len(periods) - 1):
        freq.append(periods[i+1] - periods[i]) 
    return freq

def _test_symbol_periodicity_search(sequence, prefix, pos, step):

    if sequence[pos + step] == prefix[0]:
        res = True
        print("Candidate subsequence")
        for s, j in zip(range(len(prefix)), range(pos + step, pos + step + len(prefix))):
            res = prefix[s] == sequence[j]
            if not res:
                print("Wrong candidate")
                break
        if res:
            print("Find subsequence at: {}".format(pos + step))
            return pos + step
        else:
            return -1
    else:
        return -1


def test_symbol_periodicity_search(sequence, prefix, freq):

    f = 0
    i = 0
    while i < len(sequence) - len(prefix):
        sym = sequence[i]
        if sym == prefix[0] and _search_symbols_periodicity(sequence[i:i+len(prefix)], prefix):
            res = _test_symbol_periodicity_search(sequence, prefix, i, freq[f])
            if res > 0:
                i = res
                if f < len(freq)-1:
                    f += 1
                else:
                    break
            else:
                i += 1
        else:
            i += 1

def get_community(communities, digitizer, x):
    i, _ = digitizer.get_range(x)
    for c in communities:
        if i in communities:
            return c

def periodicity_detection(data, digitizer):
    edges_from_nx = list(digitizer.graph.edges())
    graph = ig.Graph(edges=edges_from_nx, directed=True)
    communities = graph.community_infomap()

    com_visited = []
    periods = []
    com_first = get_community(communities, digitizer, data[0])

    for i in range(1, len(data)):
        com_curr = get_community(communities, digitizer, data[i])
        com_visited.append(com_curr)
        if com_curr == com_first and len(com_visited) == len(communities):
            periods.append(i)
            com_visited = []
    return periods

def polynomial_fitting(x, y, deg=2):

    coef = np.polyfit(x, y, deg)
    res = []
    exp = [i for i in range(deg+1)]
    exp.reverse()
    for val in x:
        z = 0
        for i, j in zip(range(deg+1), exp):
            z += coef[i]*(val**j)
        res.append(z)
    return res

def main(data, store, pv_name):
    lit_ts = utils.get_all_values_pv(data, pv_name)
    min_val = np.min(lit_ts)
    max_val = np.max(lit_ts)
    print("Length of input: {}, range:{}".format(len(lit_ts), (max_val-min_val)))
    pv = store[pv_name]
    d = Digitizer(max_val-min_val, min_val, max_val, nx.DiGraph())
    #d.create_graph(lit_ts)
    res = d.digitize(lit_ts)
    x_axis = np.arange(len(res))
    model = polynomial_fitting(x_axis, res)
    print("Model mean:{} , variance:{}".format(np.mean(model), np.var(model)))
    plt.plot(x_axis, res)
    plt.plot(x_axis, model)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="input")
    parser.add_argument("--conf", action="store", dest="conf")

    args = parser.parse_args()
    data = utils.read_state_file(args.input)[utils.COOL_TIME:]
    store = PVStore(args.conf)
    pv_name = "lit101"
    main(data, store, pv_name)
    pdb.set_trace()
