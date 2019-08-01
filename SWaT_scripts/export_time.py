import os
import sys
import argparse
import pickle
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ids

import ids.utils as utils

def main(data, pv, from_val, to_val, conf, output):
    var = None
    dist_thresh = 0.01
    with open(conf) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for var_desc in desc['variables']:
            if var_desc['variable']['name'] == pv:
                var = var_desc['variable']

    recorded_ts = []
    from_ts = None
    to_ts = None

    for state in data:
        val = state[pv]
        if utils.same_value(var['max'], var['min'], from_val, val, dist_thresh):
            if from_ts is None:
                from_ts = state['timestamp']
                continue
        if utils.same_value(var['max'], var['min'], to_val, val, dist_thresh):
            if from_ts is not None and to_ts is None:
                to_ts = state['timestamp']
                elapsed_time = (to_ts - from_ts).total_seconds()
                recorded_ts.append(elapsed_time)
                from_ts = None
                to_ts = None

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
