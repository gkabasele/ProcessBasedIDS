import os
import sys
import argparse
import pickle
import yaml
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ids

import ids.utils as utils

def transition_time(data, pv, max_val, min_val, from_val, to_val):
    recorded_ts = []
    from_ts = None
    to_ts = None

    for state in data:
        val = state[pv]
        if utils.same_value(max_val, min_val, from_val, val):
            if from_ts is None:
                from_ts = state['timestamp']
                continue
        if utils.same_value(max_val, min_val, to_val, val):
            if from_ts is not None and to_ts is None:
                to_ts = state['timestamp']
                elapsed_time = (to_ts - from_ts).total_seconds()
                recorded_ts.append(elapsed_time)
                from_ts = None
                to_ts = None
    return recorded_ts

def same_value_time(data, pv, max_val, min_val, value):
    recorded_ts = []
    start_ts = None
    end_ts = None

    for state in data:
        val = state[pv]
        if utils.same_value(max_val, min_val, val, value):
            if start_ts is None:
                start_ts = state['timestamp']
                end_ts = state['timestamp']
            else:
                end_ts = state['timestamp']
        else:
            if start_ts is None:
                pass
            else:
                elapsed_time = (end_ts - start_ts).total_seconds()
                recorded_ts.append(elapsed_time)
                start_ts = None
                end_ts = None
    return recorded_ts


def main(data, pv, from_val, to_val, conf, output):
    var = None
    with open(conf) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for var_desc in desc['variables']:
            if var_desc['variable']['name'] == pv:
                var = var_desc['variable']

    if to_val == from_val:
        recorded_ts = same_value_time(data, pv, var['max'], var['min'], from_val)
    else:
        recorded_ts = transition_time(data, pv, var['max'], var['min'], from_val, to_val)

    pdb.set_trace()

    with open(output, "w") as fh:
        for ts in recorded_ts:
            fh.write("{},".format(ts))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, dest="input")
    parser.add_argument("--pv", type=str, dest="pv")
    parser.add_argument("--conf", type=str, dest="conf")
    parser.add_argument("--from", type=float, dest="from_val")
    parser.add_argument("--to", type=float, dest="to_val")
    parser.add_argument("--output", type=str, dest="output")

    args = parser.parse_args()

    with open(args.input, "rb") as filename:
        data = pickle.load(open(args.input, "rb"))

    main(data, args.pv, args.from_val, args.to_val, args.conf, args.output)
