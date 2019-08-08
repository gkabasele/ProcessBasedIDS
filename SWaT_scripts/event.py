import sys
import os
import argparse
import pickle
import math
import pdb

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import readline 
import code

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ids
import ids.utils as utils

matplotlib.use('TkAgg')

EXCLUDE = ["timestamp", "normal/attack", "mv101",
           "p101", "p102", "mv201", "p201",
           "p202", "p203", "p204", "p205", "p206",
           "mv301", "mv302", "mv303",
           "mv304", "p301", "p302", "p401", "p402",
           "p403", "p404", "uv401",
           "p501", "p502", "p601",
           "p602", "p603"]

def get_readings(state, reading, exclude):

    for k, v in state.items():
        if k not in exclude:
            if k not in reading:
                reading[k] = [v]
            else:
                reading[k].append(v)

def update_min_max(state, min_values, max_values):

    for k in state:
        if k not in min_values:
            if k != "timestamp" and k != "normal/attack":
                min_values[k] = state[k]
                max_values[k] = state[k]
        else:
            min_values[k] = min(min_values[k], state[k])
            max_values[k] = max(max_values[k], state[k])

def main(filename, pv, ts):
    on_reading = {}
    off_reading = {}

    on_timestamp = []
    off_timestamp = []
    current = 0

    min_values = {}
    max_values = {}

    data = pickle.load(open(filename, "rb"))

    for i, state in enumerate(data):

        update_min_max(state, min_values, max_values)

        if i == 0:
            current = state[pv]
            print("Nbr Var:{}".format(len(state) - 2))
            print("Nbr Excluded: {}".format(len(EXCLUDE) - 2))

        else:
            if current != state[pv]:
                current = state[pv]
                if current == 1:
                    get_readings(state, off_reading, EXCLUDE)
                    off_timestamp.append(state[ts])
                elif current == 2:
                    get_readings(state, on_reading, EXCLUDE)
                    on_timestamp.append(state[ts])

    for k in min_values:
        min_val = min_values[k]
        max_val = max_values[k]
        diff = max_val - min_val
        print("Name: {}, Min:{}, Max:{}, diff: {}".format(k, min_val, max_val,
                                                          diff))
    return off_reading, on_reading

def plot_timeseries(data, pv):

    vals = np.array([x[pv] for x in data])
    x = np.array([x['timestamp'] for x in data])

    fig, ax = plt.subplots() 
    ax.set_xlabel('time(s)')
    ax.set_ylabel(pv)

    ax.plot(x, vals)

    min_val = np.min(vals)
    max_val = np.max(vals)

    #nbr_ranges = math.ceil(math.sqrt(max_val - min_val))
    #range_width = (max_val - min_val)/nbr_ranges
    #split = set()

    #for i in range(nbr_ranges):
    #    lower = min_val + i * range_width
    #    upper = min_val + (i+1) * range_width
    #    split.add(lower)
    #    split.add(upper)

    for h in np.linspace(min_val, max_val, 10):
        ax.axhline(h, color='black', lw=0.2)
    plt.show()

def plot_variable_timeseries_corr(data, act, sens):

    fig, ax1 = plt.subplots()

    length = math.floor(len(data)/2)
    x_vals = np.array([x['timestamp'] for x in data[:math.floor(len(data)/2)]])

    act_ts = []
    sens_ts = []
    for state in data[:length]:
        act_ts.append(state[act])
        sens_ts.append(state[sens])

    act_ts = np.array(act_ts)
    sens_ts = np.array(sens_ts)

    ax1.plot(x_vals, act_ts, color='b')
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel(act, color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x_vals, sens_ts, color='g')
    ax2.set_ylabel(sens, color='g')
    ax2.tick_params('y', colors='g')

    plt.show()

def y_histogram(data, pv):
    values = get_values(data, pv)
    fig, ax = plt.subplots()
    hist, bin_edges, patches = ax.hist(values, bins=100)
    pdb.set_trace()
    normalized = np.array([x/len(values) for x in hist])

    plt.show()

def plot_acorr(data, pv):
    values = get_values(data, pv)
    plot_acf(values, lags=[600, 3600, 24*3600])

    plt.title('Autocorrelation of {}'.format(pv))
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

def plot_decomp(data, pv, freq=86400):
    values = get_values(data, pv)
    result = seasonal_decompose(values, model='additive', freq=freq)
    result.plot()
    plt.show()

def plot_data(on_reading, off_reading):

    print("Nbr Sensors: {}".format(len(on_reading)))
    plt.yscale("log")

    print("Name, on: mean/std, off: mean/std")
    for key in on_reading:
        np_on_reading = np.array(on_reading[key])
        np_off_reading = np.array(off_reading[key])

        print("key: {}, on: {}/{}, off: {}/{}".format(key, np_on_reading.mean(),
                                                      np_on_reading.std(),
                                                      np_off_reading.mean(),
                                                      np_off_reading.std()))
        values = on_reading[key]
        plt.plot([key]*len(values), values, 'o')

    plt.show()

def get_values(data, pv):
    return np.array([x[pv] for x in data])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, dest="input")
    args = parser.parse_args()
    pv = "mv101"
    ts = "timestamp"
    with open(args.input, "rb") as filename:
        data = pickle.load(open(args.input, "rb"))
    variables = globals().copy()
    variables.update(locals())
    shell = code.InteractiveConsole(variables)
    shell.interact()

    """
    off_reading, on_reading = main(args.input, pv, ts)
    plot_data(on_reading, off_reading)
    """
    #plot_variable_timeseries_corr(args.input, "mv101", "mv201")
    #plot_variable_timeseries_corr(args.input, "lit101", "lit301")
    #plot_variable_timeseries_corr(args.input, "mv101", "lit101")
    #plot_variable_timeseries_corr(args.input, "mv101", "lit301")
    #plot_variable_timeseries_corr(args.input, "mv101", "lit401")
    #pdb.set_trace()
