import pdb
import argparse
from matplotlib import pyplot
import numpy as np
import utils
import string
from pvStore import PVStore

class Symbol(object):
    i = 0

    def __init__(self, val=None):
        if val is None:
            self.val = string.ascii_lowercase[Symbol.i]
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

    def __init__(self, symbol):
        self.symbol = symbol

    def same_symbol(self, other):
        return self.symbol.val == other.val

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
            symbol = self.get_symbol(value)
            res.append(SymbolEntry(symbol))

        return res

def symbol(val):
    return SymbolEntry(Symbol(val))

def test_digitize():
    digitizer = Digitizer([3, 6, 9], 0, 1)
    print(digitizer.mapping)
    input_l = [2, 5, 8, 1, 11, 10, 7, 2, 9, 13, 6, 11, 9]
    res = digitizer.digitize(input_l)

    assert [x.symbol.val for x in res] == ['a', 'c', 'e', 'a', 'g', 'g', 'e', 'a', 'f', 'g', 'd', 'g', 'f']

    prefix = [symbol('e'), symbol('a')]

    periods = search_symbols_periodicity(res, prefix) 

    print(periods)

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
    return periods

def main(data, store):
    pv_name = "lit101"
    lit_ts = utils.get_all_values_pv(data, pv_name, 86400)
    print("Length of input: {}".format(len(lit_ts)))
    pv = store[pv_name]
    digitizer = Digitizer(pv.limit_values, pv.min_val, pv.max_val)
    print(digitizer.mapping)
    res = digitizer.digitize(lit_ts)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="input")
    parser.add_argument("--conf", action="store", dest="conf")

    test_digitize()

    """
    args = parser.parse_args()
    data = utils.read_state_file(args.input)
    store = PVStore(args.conf)
    main(data, store)
    """
