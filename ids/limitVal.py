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

def moving_average(data, pv, win, limit=None):

    values, times = get_values(data, pv, limit)
    trends = []

    for i in range(len(values)):
        new_val = compute_window_average(values, i, win)
        trends.append(new_val)

    ma = np.array(trends)

    slopes = slope_graph(times, ma)

    plt.subplot(3, 1, 1)

    plt.plot(times, ma)
    plt.ylabel('moving_average_{}'.format(pv))

    plt.subplot(3, 1, 2)
    plt.plot(times[:-1], slopes)
    plt.ylabel('slopes_{}'.format(pv))

    plt.subplot(3, 1, 3)
    plt.plot(times, values)
    plt.xlabel('time(s)')
    plt.ylabel(pv)

    plt.show()


def slope_graph(times, values):
    trends = []
    for i in range(len(values)-1):
        xdiff = (times[i+1] - times[i]).total_seconds() 
        ydiff = values[i+1] - values[i]
        trends.append(ydiff/xdiff)
    slopes = np.array(trends)

    return slopes
    

def compute_b(data, n, win):
    sum_u = 0
    sum_nu = 0
    for i in range(win):
        if n + i < len(data):
            sum_u += data[n + i]
            sum_nu += i * data[n + i]

    numerator = 2*(2*win + 1)*sum_u - 6*sum_nu

    b_zero = numerator/(win*(win-1))

    numerator = 12*sum_nu - 6*(win+1)*sum_u

    b_one = numerator/(win*win*(win-1)*(win + 1))
    return b_zero, b_one

def linearize(data, pv, win, limit=None):

    if limit is None:
        values = np.array([x[pv] for x in data])
        times = np.array([x['timestamp'] for x in data])
    else:
        values = np.array([x[pv] for x in data[:limit]])
        times = np.array([x['timestamp'] for x in data[:limit]])

    trends = []

    for i in range(len(values)):
        b_zero, b_one = compute_b(values, i, win)
        trends.append(b_zero + b_one)

    linear_values = np.array(trends)

    plt.subplot(2, 1, 1)

    plt.plot(times, linear_values)
    plt.ylabel(pv)

    plt.subplot(2, 1, 2)
    plt.plot(times, values)
    plt.xlabel('time(s)')
    plt.ylabel("lin_{}".format(pv))
    plt.show()

def masks(vec):
    d = np.diff(vec)
    dd = np.diff(d)

    to_mask = ((d[:1] != 0) & (d[:-1] == -dd))
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask

def apply_mask(mask, x, y):
    return x[1:-1][mask], y[1:-1][mask]

def inflection_point(values, times):

    to_vert_mask, from_vert_mask = masks(times)
    to_horiz_mask, from_horiz_mask = masks(values)

    to_vert_t, to_vert_v = apply_mask(to_vert_mask, times, values)
    from_vert_t, from_vert_v = apply_mask(from_vert_mask, times, values)
    to_horiz_t, to_horiz_v = apply_mask(to_horiz_mask, times, values)
    from_horiz_t, from_horiz_v = apply_mask(from_horiz_mask, times, values)

    plt.plot(times, values, 'b-')
    plt.plot(to_vert_t, to_vert_v, 'r^', label='Plot goes vertical')
    plt.plot(from_vert_t, from_vert_v, 'kv', label='Plot stops being vertical')
    plt.plot(to_horiz_t, to_horiz_v, 'r>', label='Plot goes horizontal')
    plt.plot(from_horiz_t, from_horiz_v, 'k<', label='Plot stops being horizontal')

    plt.legend()
    plt.show()

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
