import pickle
import argparse
import pdb

from scipy.signal import savgol_filter, argrelextrema
import numpy as np

from timePattern import find_extreme_local
from utils import *
from welford import Welford

import predicate as pd

"""
Approach that take advantage of the fact that data point tend to group
around critical value as the slope will change. Drastic increase or decrease
take some time so those critical values appear more often. Histogram from 
y-axis
"""

IGNORE_COL = ["timestamp", "normal/attack"]

#THRESH = 0.2
THRESH = 1
# 0.01, 0.05 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1

def fill_map_from_ts(state, actuators, map_var_event_val, event):
    for k, v in state.items():
        if k not in actuators and k not in IGNORE_COL:
            if map_var_event_val[k] is None:
                map_var_event_val[k] = {event : list()}
            elif event not in map_var_event_val[k]:
                map_var_event_val[k][event] = list()

            map_var_event_val[k][event].append(v)

def get_values_and_event_values(data, actuators, map_var_event_val, events):
    # map_var_event_val: {var1: {Event1: [val], Event2: [val],...}, }
    # events: {var1 : {ON: Event, OFF: Event}, var2: ...}
    for types in events.values():
        for event in types.values():
            for ts in event.timestamps:
                state = data[ts]
                fill_map_from_ts(state, actuators, map_var_event_val, event)

def eligible_critical(max_val, min_val, val):
    return (same_value(max_val, min_val, val.mean, val.mean + val.std) and
            same_value(max_val, min_val, val.mean, val.mean - val.std))

            

## Smoothing approach to find critical ##

def find_extreme_from_smooth_data(name, values, event_values, inputtype,
                                  windows=101, order=3):
    min_val = min(values)
    max_val = max(values)
    filtered_event_values = [x for x in event_values if eligible_critical(max_val, min_val, x)]

    crit_vals = []
    if inputtype == "cont":
        smooth_vals = savgol_filter(values, windows, order)
        min_pos = argrelextrema(smooth_vals, np.less)
        max_pos = argrelextrema(smooth_vals, np.greater)

        extremes = [values[i] for i in min_pos]
        extremes.extend([values[i] for i in max_pos])

        extremes_list = np.concatenate(extremes).ravel()

        pdb.set_trace()
        ###
        hist, bin_edges = np.histogram(extremes_list, bins=10)

        ranges = []
        total = sum(hist)
        if total == 0:
            return []

        for i in range(len(bin_edges) - 1):
            lower = bin_edges[i]
            upper = bin_edges[i+1]
            count = hist[i]
            norm = hist[i]/total
            if norm >= THRESH:
                ranges.append(RangeVal(lower, upper, count, norm))

        blocks = merge_ranges(ranges, len(values))
        return blocks

    else:
        smooth_vals = values
        min_pos, max_pos = find_extreme_local(smooth_vals)

        extremes = [values[i] for i in min_pos]
        extremes.extend([values[i] for i in max_pos])

        extremes_list = extremes

        hist, bin_edges = np.histogram(extremes_list, bins=10)

        ranges = []
        total = sum(hist)
        if total == 0:
            return []

        for i in range(len(bin_edges) - 1):
            lower = bin_edges[i]
            upper = bin_edges[i+1]
            count = hist[i]
            norm = hist[i]/total
            if norm >= THRESH:
                ranges.append(RangeVal(lower, upper, count, norm))

        blocks = merge_ranges(ranges, len(values))
        return blocks


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
        if norm >= THRESH:
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

def get_actuators(conf_file):
    with open(conf_file) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        actuators = list()
        for variable in desc['variables']:
            var = variable['variable']
            if var["type"] == "co":
                actuators.append(var['name'])
        return actuators

def get_actuators_sensors(conf_file):
    with open(conf_file) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        actuators = list()
        sensors = list()
        for variable in desc["variables"]:
            var = variable["variable"]
            if var["type"] == "co":
                actuators.append(var["name"])
            elif var["type"] == "hr":
                sensors.append(var["name"])

        return actuators, sensors

def main(data, conf, output, strategy, cool_time, inputtype):
    actuators = get_actuators(conf)
    events = pd.retrieve_update_timestamps(actuators, data)
    map_var_event_val = {x:None for x in data[0].keys() if x not in actuators and x not in IGNORE_COL}
    get_values_and_event_values(data, actuators, map_var_event_val, events)
    pdb.set_trace()
    with open(conf) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for variable in desc['variables']:
            var = variable['variable']
            values, _ = get_values(data, var["name"])
            if var["type"] == "hr" or var["type"] == "ir":
                min_val = np.min(values)
                max_val = np.max(values)

                event_values = list()
                for event, group in map_var_event_val[var["name"]].items():
                    event_values.append(group)

                blocks = find_extreme_from_smooth_data(var["name"], values,
                                                       event_values,
                                                       inputtype)
                vals = []
                for block in blocks:
                    vals.append(block.lower.item())
                    vals.append(block.upper.item())

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

    if args.inputtype == "cont":
       main(data, args.conf, args.output, args.strategy, args.cool_time, args.inputtype)
    else:
       main(data, args.conf, args.output, args.strategy, args.cool_time, args.inputtype)
