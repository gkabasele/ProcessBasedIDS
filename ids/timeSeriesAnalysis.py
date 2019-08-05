import pdb
import math
import argparse
from matplotlib import pyplot
import numpy as np
import utils
import string
from scipy import stats
from pvStore import PVStore

ALPHA = string.ascii_lowercase

class Graph(object):

    def __init__(self):
        self.graph = {}

    def add_edge(self, node, neigh):
        if node not in self.graph:
            self.graph[node] = set([neigh])
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
    j = 0

    def __init__(self, val=None):
        if val is None:
            s = ALPHA[Symbol.i]
            k = 0
            for _ in range(Symbol.j):
                s += ALPHA[k]
                k += 1
            self.val = s
            Symbol.i += 1
            if Symbol.i == len(ALPHA):
                Symbol.j += 1
                Symbol.i = 0
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

    def __init__(self, bins, min_val, max_val):
        self.bins = bins
        self.mapping = {}
        self.min_val = min_val
        self.max_val = max_val
        for i, val in enumerate(bins):
            if i == 0:
                self.mapping["<{}".format(val)] = Symbol()

            self.mapping[str(val)] = Symbol()

            if i < len(bins)-1:
                self.mapping["[{},{}]".format(val, bins[i+1])] = Symbol()
            else:
                self.mapping[">{}".format(val)] = Symbol()

    def get_symbol(self, value):
        for i, v in enumerate(self.bins):
            if utils.same_value(self.max_val, self.min_val, v, value):
                return self.mapping[str(v)]

            elif value < v:
                if i == 0:
                    return self.mapping["<{}".format(v)]
                else:
                    return self.mapping["[{},{}]".format(self.bins[i-1], v)]
            else:
                if i == len(self.bins)-1:
                    return self.mapping[">{}".format(v)]

    def digitize(self, data):
        res = []

        for i, value in enumerate(data):
            sym = self.get_symbol(value)
            if len(res) == 0:
                res.append(SymbolEntry(sym, i))
            else:
                last_symbol = res[-1]
                if last_symbol.same_symbol(sym):
                    last_symbol.end_time = i
                else:
                    res.append(SymbolEntry(sym, i))
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

def create_graph(res):
    g = Graph()
    for i in range(len(res) - 1):
        g.add_edge(res[i].val(), res[i+1].val())

    return g

def _periodicity_test(partition, cont_vars_part):

    for state in partition:
        for k in cont_vars_part:
            cont_vars_part[k].append(state[k])

def periodicity_test(data, store, slen, thresh=utils.DIST):

    cont_vars = store.continous_vars()
    cont_vars_part = {x: [] for x in cont_vars}
    vars_means = {x: [] for x in cont_vars}
    nbr_partition = math.ceil(len(data)/slen)

    for i in range(nbr_partition):

        _periodicity_test(data[i*slen:(i+1)*slen], cont_vars_part)

        for k, v in cont_vars_part.items():
            vars_means[k].append(np.mean(v))
            cont_vars_part[k] = []

    for k, v in vars_means.items():
        var = store[k]
        diff_means = []
        for i in range(len(v) - 1):
            for j in range(i+1, len(v)):
                diff = math.fabs(v[i] - v[j])
                diff_means.append(diff)
        same_dist_test = (var.max_val - var.min_val)*thresh
        if np.mean(diff_means) <= same_dist_test:
            print("Name:{}".format(k))

def partition_statistic(data, store, pv_name, slen, thresh=utils.DIST):

    vals = utils.get_all_values_pv(data, pv_name)
    nbr_partition = math.ceil(len(vals)/slen)

    means = []
    var_s = []

    for i in range(nbr_partition):
        partition = vals[i*slen:(i+1)*slen]
        means.append(np.mean(partition))
        var_s.append(np.var(partition))

    print(means)

    diff_means = []

    for i in range(len(means) - 1):
        for j in range(i+1, len(means)):
            diff = math.fabs(means[i] - means[j])
            diff_means.append(diff)

    var = store[pv_name]
    same_dist_test = (var.max_val - var.min_val)*thresh
    print("Name:{}".format(pv_name))
    print("MeanDiff:{}, {}".format(np.mean(diff_means), same_dist_test))

def main(data, store):
    pv_name = "lit101"
    lit_ts = utils.get_all_values_pv(data, pv_name, utils.DAY_IN_SEC)
    print("Length of input: {}".format(len(lit_ts)))
    pv = store[pv_name]
    digitizer = Digitizer(pv.limit_values, pv.min_val, pv.max_val)
    print(digitizer.mapping)
    res = digitizer.digitize(lit_ts)
    print(res)

    """
    g = create_graph(res)
    print(g.find_path("b", "h"))
    print(g.find_path("h", "b"))
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="input")
    parser.add_argument("--conf", action="store", dest="conf")

    args = parser.parse_args()
    data = utils.read_state_file(args.input)[utils.COOL_TIME:]
    store = PVStore(args.conf)

    #partition_statistic(data, store, "lit101", utils.DAY_IN_SEC)
    periodicity_test(data, store, utils.DAY_IN_SEC)

    """
    main(data, store)
    """
