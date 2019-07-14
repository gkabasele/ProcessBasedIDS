import pickle
import argparse
import readline
import code
import pdb
import math
from datetime import datetime, timedelta

import numpy as np
import pprint
import matplotlib
import matplotlib.pyplot as plt

TRESH = 0.01

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

def compute_ranges(values):
    hist, bin_edges = np.histogram(values, bins=100)

    ranges = []
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        count = hist[i]
        norm = hist[i]/len(values)
        if norm >= TRESH:
            ranges.append(RangeVal(lower, upper, count, norm))
    return ranges

def merge_ranges(ranges, number):

    blocks = []
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

def divide_and_conquer(data, pv, limit=None):
    values, _ = get_values(data, pv, limit)

    ranges = compute_ranges(values)
    pdb.set_trace()
    blocks = merge_ranges(ranges, len(values))



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

def moving_average(values, win):
    trends = []
    for i in range(len(values)):
        new_val = compute_window_average(values, i, win)
        trends.append(new_val)
    return np.array(trends)

def limit_values(data, pv, win, limit=None):

    values, times = get_values(data, pv, limit)

    seconds = np.arange(len(times))
    
    ma = moving_average(values, win)

    slopes = moving_average(slope_graph(seconds, ma), win)
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
    

def main(data, pv):
    values = np.array([x[pv] for x in data])
    hist, bin_edges = np.histogram(values, bins=100, density=True)

    dist = {}

    for i, edges in enumerate(bin_edges[:-1]):
        dist[edges] = hist[i]

    pprint.pprint(dist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="filename", action="store")
    args = parser.parse_args()
    with open(args.filename, "rb") as filename:
        data = pickle.load(filename)
    variables = globals().copy()
    variables.update(locals())
    shell = code.InteractiveConsole(variables)
    shell.interact()
