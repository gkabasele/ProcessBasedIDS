import os 
import sys
import argparse
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from welford import Welford

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, dest="input")
    args = parser.parse_args()
    pv = "mv101"
    ts = "timestamp"

    main(args.input, pv, ts)
