import pickle
import argparse
import readline
import code
import pdb
import math
from datetime import datetime, timedelta

from scipy.signal import savgol_filter, argrelextrema
import numpy as np
import pprint
import matplotlib
import matplotlib.pyplot as plt

from utils import *

"""
Approach that take advantage of the fact that data point tend to group
around critical value as the slope will change. Drastic increase or decrease
take some time so those critical values appear more often. Histogram from 
y-axis
"""

THRESH = 0.2

class RangeVal(object):

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

## Simple splitting ##

def split_values_spectre(values, n_split=4):
    min_val = float(min(values))
    max_val = float(max(values))
    spectre = max_val - min_val
    gap = float(spectre)/n_split
    vals = list()
    curr = min_val
    while curr <= max_val:
        vals.append(curr)
        curr += gap
    return vals

## Smoothing approach to find critical ##

def find_extreme_from_smooth_data(values, windows=1001, order=3):
    smooth_vals = savgol_filter(values, windows, order)
    min_pos = argrelextrema(smooth_vals, np.less)
    max_pos = argrelextrema(smooth_vals, np.greater)

    extremes = [values[i] for i in min_pos]
    extremes.extend([values[i] for i in max_pos])

    extremes_list = np.concatenate(extremes).ravel()

    hist, bin_edges = np.histogram(extremes_list, bins=10)

    ranges = []
    total = sum(hist)
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        count = hist[i]
        norm = hist[i]/total
        if norm >= THRESH:
            ranges.append(RangeVal(lower, upper, count, norm))

    blocks = merge_ranges(ranges, len(values))
    return blocks


def find_inflection_point(data):
    maximas = list()
    minimas = list()

    for i in range(len(data)):
        if i == 0:
            if data[i] > data[i+1]:
                maximas.append(i)
            elif data[i] < data[i+1]:
                minimas.append(i)
        elif i == len(data)-1:
            if data[i] > data[i-1]:
                maximas.append(i)
            elif data[i] < data[i-1]:
                minimas.append(i)
        else:
            if data[i-1] < data[i] and data[i+1] < data[i]:
                maximas.append(i)
            elif data[i-1] > data[i] and data[i+1] > data[i]:
                minimas.append(i)
    return maximas, minimas

def compute_ranges(values, inputtype):
    if inputtype == "cont":
        hist, bin_edges = np.histogram(values, bins=100)

    elif inputtype == "disc":
        min_val = np.min(values)
        max_val = np.max(values)
        hist, bin_edges = np.histogram(values, bins=max_val - min_val)

    ranges = []
    total = sum(hist)
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        count = hist[i]
        norm = hist[i]/total
        if norm >= TRESH:
            ranges.append(RangeVal(lower, upper, count, norm))
    return ranges

def merge_ranges(ranges, number):

    blocks = []
    if len(ranges) <= 1:
        return ranges 

    start_block = ranges[0].lower
    end_block = ranges[0].upper
    count = 0
    for i in range(len(ranges)-1):
        block = ranges[i]
        next_block = ranges[i+1]

        if block.upper != next_block.lower:
            count += block.count
            r = RangeVal(start_block, end_block, count)
            r.normalized(number)
            blocks.append(r)

            start_block = next_block.lower
            end_block = next_block.upper

        else:
            end_block = next_block.upper
            count += block.count

        if i == len(ranges) - 2:

            if block.upper == next_block.lower:
                count += block.count + next_block.count
                r = RangeVal(start_block, next_block.upper, count)
                r.normalized(number)
                blocks.append(r)
            else:
                blocks.append(next_block)

    return blocks

def find_limit_values(values):
    xs, y_data = compute_kde(values)
    maxima_indices, minima_indices = find_inflection_point(y_data)
    minimas = [xs[i] for i in minima_indices]
    minimas.extend([xs[i] for i in maxima_indices])
    minimas.sort()
    return minimas

def divide_and_conquer(values, inputtype):
    ranges = compute_ranges(values, inputtype)
    blocks = merge_ranges(ranges, len(values))
    return blocks

def get_values(data, pv, limit=None):

    if limit is None:
        values = np.array([x[pv] for x in data])
        times = np.array([x['timestamp'] for x in data])
    else:
        values = np.array([x[pv] for x in data[:limit]])
        times = np.array([x['timestamp'] for x in data[:limit]])

    return values, times

def compute_window_average(data, i, win):
    low = max(0, i - win)
    high = min(len(data)-1, i + win)
    mv_avg = sum(data[low:high])/(high - low)
    return mv_avg

def compute_trends(values, win):
    trends = []
    for i in range(len(values)):
        new_val = compute_window_average(values, i, win)
        trends.append(new_val)
    return np.array(trends)

def limit_values(data, pv, win, limit=None):

    values, times = get_values(data, pv, limit)

    seconds = np.arange(len(times))
    
    ma = compute_trends(values, win)

    slopes = compute_trends(slope_graph(seconds, ma), win)
    change_points = np.diff(slopes)

    plt.subplot(4, 1, 1)
    plt.plot(times, values)
    plt.ylabel(pv)

    plt.subplot(4, 1, 2)
    plt.plot(times, ma)
    plt.ylabel('moving_average')

    plt.subplot(4, 1, 3)
    plt.plot(times[:-1], slopes)
    plt.ylabel('Ft Deri')

    plt.subplot(4, 1, 4)
    plt.plot(times[:-2], change_points)
    plt.ylabel('Sd Deri')
    plt.xlabel('time(s)')

    plt.show()


def slope_graph(times, values):
    trends = []
    for i in range(len(values)-1):
        xdiff = times[i+1] - times[i]
        ydiff = values[i+1] - values[i]
        trends.append(ydiff/xdiff)
    slopes = np.array(trends)

    return slopes

def main(data, conf, output, strategy, cool_time, inputtype):
    with open(conf) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for variable in desc['variables']:
            var = variable['variable']
            if var["type"] == "hr" or var["type"] == "ir":
                values, _ = get_values(data, var['name'])
                min_val = np.min(values)
                max_val = np.max(values)
                if strategy == "simple":
                    vals = split_values_spectre(values[cool_time:])

                elif strategy == "smooth":
                    blocks = find_extreme_from_smooth_data(values[cool_time:])
                    vals = []
                    for block in blocks:
                        vals.append(block.lower.item())
                        vals.append(block.upper.item())

                elif strategy == "hist":
                    blocks = divide_and_conquer(values[cool_time:], inputtype)
                    vals = []
                    for block in blocks:
                        vals.append(block.lower.item())
                        vals.append(block.upper.item())
                elif strategy == "kde":
                    vals = [float(x) for x in find_limit_values(values[cool_time:])]

                var['critical'] = vals
                var['min'] = float(min_val)
                var['max'] = float(max_val)

        with open(output, "w") as ofh:
            content = yaml.dump(desc, default_flow_style=False)
            ofh.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="filename", action="store")
    parser.add_argument("--conf", dest="conf", action="store")
    parser.add_argument("--output", dest="output", action="store")
    parser.add_argument("--strategy", default="all", choices=["kde", "hist", "simple", "smooth"])
    parser.add_argument("--cool", default=COOL_TIME, type=int, dest="cool_time")
    parser.add_argument("--inputtype", default="cont", choices=["cont", "disc"])
    args = parser.parse_args()

    with open(args.filename, "rb") as filename:
        data = pickle.load(filename)

    main(data, args.conf, args.output, args.strategy, args.cool_time, args.inputtype)
    """
    variables = globals().copy()
    variables.update(locals())
    shell = code.InteractiveConsole(variables)
    shell.interact()
    """
