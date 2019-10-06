import pdb
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
import utils
import string
from scipy import stats
from limitVal import RangeVal

NBR_RANGE = 10

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

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.ranges = self.compute_ranges()
        self.res = list()

    def compute_ranges(self):
        ranges = list()
        nbr_ranges = NBR_RANGE
        ranges_width = (self.max_val - self.min_val)/nbr_ranges

        for i in range(nbr_ranges):
            lower = self.min_val + i * ranges_width
            upper = self.min_val + (i+1)*ranges_width
            r = RangeVal(lower, upper, 0)
            ranges.append(r)

        return ranges

    def get_range(self, x):
        if x <= self.ranges[0].lower:
            return 0, self.ranges[0]

        if x >= self.ranges[-1].upper:
            return len(self.ranges)-1, self.ranges[-1]

        for i, rangeval in enumerate(self.ranges):
            if x >= rangeval.lower and x <= rangeval.upper:
                return i, rangeval

    def online_digitize(self, x):
        i, _ = self.get_range(x)
        self.res.append(i)

    def digitize(self, data):
        res = list()
        for val in data:
            i, _ = self.get_range(val)
            res.append(i)
        return res

def symbol(val):
    return SymbolEntry(Symbol(val))

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

def main(data, pv_name, training_size):
    lit_ts = utils.get_all_values_pv(data, pv_name)[:training_size]
    min_val = np.min(lit_ts)
    max_val = np.max(lit_ts)
    print("Length of input: {}, range:{}".format(len(lit_ts), (max_val-min_val)))
    d = Digitizer(min_val, max_val)
    res = d.digitize(lit_ts)
    x_axis = np.arange(len(res))
    model = polynomial_fitting(x_axis, res)
    print("Model mean:{} , variance:{}".format(np.mean(model), np.var(model)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="input")
    parser.add_argument("--size", type=int, default=utils.DAY_IN_SEC, action="store", dest="training_size",
                        help="number of sample to consider")
    parser.add_argument("--cool", type=int, default=utils.COOL_TIME, action="store", dest="cool")

    args = parser.parse_args()
    data = utils.read_state_file(args.input)[args.cool:]
    pv_name = "dpit301"
    main(data,pv_name, args.training_size)
    pdb.set_trace()
