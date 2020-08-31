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

class Digitizer(object):

    def __init__(self, min_val, max_val, nbr_range=NBR_RANGE):
        self.min_val = min_val
        self.max_val = max_val
        #self.ranges = self.compute_ranges(nbr_range)
        self.nbr_range = nbr_range
        self.width = (self.max_val - self.min_val)/nbr_range
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

    def get_range(self, x, debug):

        if debug:
            pdb.set_trace()

        if x <= self.min_val:
            return 0, (self.min_val, self.min_val + self.width)

        if x >= self.max_val:
            return self.nbr_range-1, (self.max_val - self.width, self.max_val)


        float_i = abs(x-self.min_val)/self.width
        i = math.floor(float_i)

        # if x is on a limit, we consider is in the zone just before
        if i != 0 and float_i % self.width == 0:
            lower = self.min_val + (i-1)*self.width
            return i-1, (lower, lower + self.width)

        lower = self.min_val + i*self.width

        return i, (lower, lower + self.width)

    def get_range_extreme(self, i):
        lower = self.min_val + i*self.width
        upper = lower + self.width
        return lower, upper

    def online_digitize(self, x):
        i, _ = self.get_range(x)
        self.res.append(i)

    def digitize(self, data):
        res = list()
        for val in data:
            i, _ = self.get_range(val)
            res.append(i)
        return res

    def convert_digitizer_index(self, other_digitizer, i, debug):
        if debug:
            pdb.set_trace()

        other_lower, other_upper = other_digitizer.get_range_extreme(i)
        i_lower, _ = self.get_range(other_lower, debug)
        i_upper, _ = self.get_range(other_upper, debug)

        if i_lower == i_upper:
            return i_lower

        # Getting the intersection of the bins
        _, lower_bin_u = self.get_range_extreme(i_lower)
        upper_bin_l, _ = self.get_range_extreme(i_upper)

        n = abs(lower_bin_u - other_lower)
        m = abs(other_upper - upper_bin_l)

        if n >= m:
            return i_lower
        else:
            return i_upper


    def __str__(self):
        return str("(Min:{},Max:{},#width:{}".format(self.min_val, self.max_val, self.width))

    def __repr__(self):
        return str(self)


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

def _test_digitizer(d, val, res, res_ranges, debug=False):

    print("get_range({})".format(val))
    i, ranges = d.get_range(val, debug)

    try:
        assert i == res
    except AssertionError:
        print("Expected:{}, got:{}".format(res, i))

    try:
        assert ranges[0] == res_ranges[0] and ranges[1] == res_ranges[1]
    except AssertionError:
        print("Expected:{}, got:{}".format(ranges, res_ranges))

def test_digitizer():

    d = Digitizer(0, 40, 8)

    _test_digitizer(d, 5, 0, (0, 5))

    _test_digitizer(d, 0, 0, (0, 5))

    _test_digitizer(d, 45, 7, (35, 40))

    _test_digitizer(d, 17, 3, (15, 20))

    d = Digitizer(40.5, 70.3, 10)

    _test_digitizer(d, 40.5, 0, (40.5, 43.48))

    _test_digitizer(d, 70.3, 9, (67.32, 70.3))

    _test_digitizer(d, 51.7, 3, (49.44, 52.42))

    _test_digitizer(d, 66.83, 8, (64.34, 67.32))

    _test_digitizer(d, 0, 0, (40.5, 43.48))

    _test_digitizer(d, 71, 9, (67.32, 70.3))

    # dist = 2,98

    d = Digitizer(-10, 30, 20)

    _test_digitizer(d, -3, 3, (-4, -2))

def _test_convert(big, small, i, res, debug=False):

    print("convert({})".format(i))
    try:
        new_index = big.convert_digitizer_index(small, i, debug)
        assert new_index == res
    except AssertionError:
        print("Expected: {}, got: {}".format(res, new_index))

def test_convert():

    dig_big = Digitizer(0, 40, 8)
    dig_small = Digitizer(0, 40, 16)

    _test_convert(dig_big, dig_small, 10, 5)
    _test_convert(dig_big, dig_small, 11, 5)
    _test_convert(dig_big, dig_small, 0, 0)
    _test_convert(dig_big, dig_small, 15, 7)

    #--------------------

    dig_small = Digitizer(0, 40, 13)

    _test_convert(dig_big, dig_small, 4, 2)
    _test_convert(dig_big, dig_small, 1, 0)
    _test_convert(dig_big, dig_small, 7, 4)
    _test_convert(dig_big, dig_small, 0, 0)
    _test_convert(dig_big, dig_small, 12, 7)

    #---------------------
    dig_big = Digitizer(-8, 16, 8)
    dig_small = Digitizer(-8, 16, 12)
    _test_convert(dig_big, dig_small, 4, 2)
    _test_convert(dig_big, dig_small, 9, 6)
    _test_convert(dig_big, dig_small, 1, 0)
    _test_convert(dig_big, dig_small, 11, 7)
    _test_convert(dig_big, dig_small, 2, 1)

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
    test_convert()

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
