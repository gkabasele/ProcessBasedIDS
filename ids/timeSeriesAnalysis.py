import pdb
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import utils
import string
from scipy import stats

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

    def __init__(self, min_val, max_val, nbr_range=NBR_RANGE):
        self.min_val = min_val
        self.max_val = max_val
        self.ranges = self.compute_ranges(nbr_range)
        self.res = list()

    def compute_ranges(self, nbr_range):
        ranges = list()
        ranges_width = (self.max_val - self.min_val)/nbr_range

        for i in range(nbr_range):
            lower = self.min_val + i * ranges_width
            upper = self.min_val + (i+1)*ranges_width
            r = utils.RangeVal(lower, upper, 0)
            ranges.append(r)

        return ranges

    def get_range(self, x):
        if x <= self.ranges[0].lower:
            return 0, self.ranges[0]

        if x >= self.ranges[-1].upper:
            return len(self.ranges)-1, self.ranges[-1]

        dist = self.ranges[0].upper - self.ranges[0].lower
        i = math.floor((x-self.ranges[0].lower)/dist)

        if x == self.ranges[i].lower:
            return i-1, self.ranges[i-1]

        return i, self.ranges[i]

        
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

def _test_digitizer(d, val, res):

    print("get_range({})".format(val))
    i, _ = d.get_range(val)

    try:
        assert i == res
    except AssertionError:
        print("Expected:{}, got:{}".format(res, i))

def test_digitizer():

    d = Digitizer(0, 40, 8)

    print(d.ranges)

    try:
        assert len(d.ranges) == 8
    except AssertionError:
        print("len(d.ranges")
        print("Expected:8, got: {}".format(len(d.ranges)))


    _test_digitizer(d, 5, 0)

    _test_digitizer(d, 0,0) 

    _test_digitizer(d, 45, 7)
    
    _test_digitizer(d, 17, 3) 

    d = Digitizer(40.5, 70.3, 10)

    print(d.ranges)

    _test_digitizer(d, 40.5, 0)

    _test_digitizer(d, 70.3, 9)

    _test_digitizer(d, 51.7, 3)

    _test_digitizer(d, 66.83, 8)

    _test_digitizer(d, 0, 0)

    _test_digitizer(d, 71, 9)

    # dist = 2,98


def main(input_data, pv_name_period, pv_name_non_period):
    data = input_data[:86401] 
    lit_ts = utils.get_all_values_pv(data, pv_name_period)
    ait_ts = utils.get_all_values_pv(data, pv_name_non_period)

    ts = [state["timestamp"] for state in data]

    min_val = np.min(lit_ts)
    max_val = np.max(lit_ts)

    min_val_non_per = np.min(ait_ts)
    max_val_non_per = np.max(ait_ts)

    print("Length of input: {}, range:{}".format(len(lit_ts), (max_val-min_val)))
    d = Digitizer(min_val, max_val)
    res = d.digitize(lit_ts)
    x_axis = np.arange(len(res))
    model = polynomial_fitting(x_axis, res)
    print("Model mean:{}, variance:{}".format(np.mean(model), np.var(model)))

    d_np = Digitizer(min_val_non_per, max_val_non_per)
    res_np = d_np.digitize(ait_ts)
    model_np = polynomial_fitting(x_axis, res_np)
    print("Model mean:{}, variance:{}".format(np.mean(model_np), np.var(model_np)))

    fig, axs = plt.subplots(2)
    axs[0].plot(ts, res)
    axs[0].plot(ts, model)

    axs[1].plot(ts, res_np)
    axs[1].plot(ts, model_np)

    plt.show()

if __name__ == "__main__":

    test_digitizer()

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="input")
    parser.add_argument("--cool", type=int, default=0, action="store", dest="cool")

    args = parser.parse_args()
    data = utils.read_state_file(args.input)[args.cool:]
    pv_name_period = "lit101"
    pv_name_non_period = "ait402"
    main(data, pv_name_period, pv_name_non_period)
    pdb.set_trace()
    """
