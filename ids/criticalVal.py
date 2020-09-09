import pdb
import math
import argparse
from collections import Counter

import yaml
import pprint
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from utils import *
import predicate as pd
import limitVal
from welford import Welford
from timeSeriesAnalysis import Digitizer

"""
J-Measure approach. Looking for values occuring when one of the
actuators is changing. Taking into account: probability of actuators changing,
probability of sensors values and conditional probability
"""

DIST = 0.01

MAGNITUDE = 0.1

TRESH = 0.0001

MAX_RANGE = 100

def get_all_values(data):
    map_var_val = {x: list() for x in data[0].keys()}

    for state in data:
        for k, v in state.items():
            map_var_val[k].append(state[k])

    return map_var_val

def get_min_max_values(data, sensors):

    map_var_val = {x: None for x in sensors}

    for state in data:
        for k, v in map_var_val.items():
            if v is None:
                map_var_val[k] = (state[k], state[k])
            else:
                map_var_val[k] = (min(state[k], v[0]), max(state[k], v[1]))

    return map_var_val

def filter_data(data, actuators, windows=101, order=3):
    map_var_val = {x: list() for x in data[0].keys() if x not in actuators and x not in limitVal.IGNORE_COL}
    for state in data:
        for k, v in state.items():
            if k in map_var_val:
                map_var_val[k].append(state[k])

    for k in map_var_val:
        map_var_val[k] = np.array(savgol_filter(map_var_val[k], windows, order))

    data_filtered = list()

    for i, state in enumerate(data):
        new_state = dict()
        for act in actuators:
            new_state[act] = state[act]
        for var in limitVal.IGNORE_COL:
            new_state[var] = state[var]
        for var in map_var_val:
            new_state[var] = map_var_val[var][i]

        data_filtered.append(new_state)

    return data_filtered


def get_most_probable_range(values):
    c = Counter(values)
    total = sum(c.values())
    max_prob = -1
    max_range = None
    for k in c:
        prob = c[k]/total

        if prob > max_prob:
            max_prob = prob
            max_range = k

    return max_range, max_prob

def get_cand_critical_values_from_std(data, actuators, sensors):

    events = pd.retrieve_update_timestamps(actuators, data)
    var_min_max = get_min_max_values(data, sensors)
    var_max_split = dict()
    event_var_critical = dict()
    event_var_ratio = dict()

    for event_name in events:
        for event in events[event_name].values():
            print("Starting event: {}".format(event))

            if len(event.timestamps) < 3:
                continue

            for var in sensors:
                event_var_values = [data[ts][var] for ts in event.timestamps]
                event_std = np.std(event_var_values)
                sigma_mul = 6

                # Six Sigma rule
                while True:
                    nbr_range = max(1, math.floor((var_min_max[var][1] - var_min_max[var][0])/(sigma_mul*event_std)))

                    if nbr_range <= 1:
                        break

                    digitizer = Digitizer(var_min_max[var][0], var_min_max[var][1], nbr_range)

                    dis_var_val = [digitizer.get_range(x)[0] for x in event_var_values]

                    range_index, prob = get_most_probable_range(dis_var_val)

                    if prob >= 1:

                        if event not in event_var_critical:
                            event_var_critical[event] = dict()

                        if var not in event_var_ratio:
                            event_var_ratio[var] = dict()


                        if var not in var_max_split:
                            var_max_split[var] = nbr_range

                        event_var_critical[event][var] = (digitizer.min_val,
                                                          digitizer.max_val,
                                                          nbr_range, range_index)

                        width_norm = digitizer.width/(var_min_max[var][1] - var_min_max[var][0])

                        event_var_ratio[var][event] = [width_norm, prob]

                        break

                    sigma_mul += 1

    return event_var_critical, event_var_ratio

def filter_based_on_range(data, event_var_critical, var_max_split, event_var_ratio):
    event_var_to_remove = {x: set() for x in event_var_critical}

    var_min_split = dict()

    # Filter the variable where noise seems inconsistent
    for event, variables in event_var_critical.items():
        for var, split in variables.items():
            _, _, nbr_range, _ = split
            max_split = var_max_split[var]

            if nbr_range/max_split < 0.5:
                event_var_to_remove[event].add(var)

        for event, remove_set in event_var_to_remove.items():
            for var in remove_set:
                event_var_critical[event].pop(var, None)

    # Retrieve min
    for event, variables in event_var_critical.items():
        for var, split in variables.items():
            _, _, nbr_range, _ = split
            if var in var_min_split:
                var_min_split[var] = min(var_min_split[var], nbr_range)
            else:
                var_min_split[var] = nbr_range

    return var_min_split

def get_all_ratio(event_var_ratio):

    X = list()

    for _, events in event_var_ratio.items():
        for value in events.values():
            X.append(value)

    return np.array(X)

def get_clusters_closest_to_zero(clusters):
    min_label = None
    min_score = 2

    for label, centroid in enumerate(clusters):
        width, prob = centroid
        score = width/prob
        if score < min_score:
            min_score = score
            min_label = label

    return min_label

def filter_based_on_ratio(data, event_var_critical, event_var_ratio):

    event_var_to_remove = {x: set() for x in event_var_critical}

    array = get_all_ratio(event_var_ratio)

    clusters = MeanShift().fit(array)

    min_label = get_clusters_closest_to_zero(clusters.cluster_centers_)

    for event, variables in event_var_critical.items():
        for var in variables:
            if clusters.predict([event_var_ratio[var][event]])[0] != min_label:
                event_var_to_remove[event].add(var)

    for event, remove_set in event_var_to_remove.items():
        for var in remove_set:
            event_var_critical[event].pop(var, None)

def get_max_split_per_var(event_var_critical):

    var_min_max = dict()

    for _, variables in event_var_critical.items():
        for var, values in variables.items():
            _, _, nbr_range, _ = values
            if var in var_min_max:
                var_min_max[var] = (min(nbr_range, var_min_max[var][0]),
                                    max(min(nbr_range, MAX_RANGE), var_min_max[var][1]))
            else:
                var_min_max[var] = (nbr_range, min(nbr_range, MAX_RANGE))

    return var_min_max

def get_var_to_critical_value(event_var_critical, var_max_split):
    # var -> nbr_range, [val1, val2, val3]
    var_to_crit = {x:set() for x in var_max_split.keys()}
    var_to_digitizer = dict()
    for _, variables in event_var_critical.items():
        for var, split in variables.items():
            min_val, max_val, nbr_range, i = split

            if var in var_to_digitizer:
                max_digit = var_to_digitizer[var]
            else:
                var_to_digitizer[var] = Digitizer(min_val, max_val, var_max_split[var][1])
                max_digit = var_to_digitizer[var]
                var_to_crit[var].add(max_digit.get_range(min_val)[0])
                var_to_crit[var].add(max_digit.get_range(max_val)[0])

            event_digit = Digitizer(min_val, max_val, nbr_range)
            new_index = max_digit.convert_digitizer_index(event_digit, i)

            if isinstance(new_index, list):
                for v in new_index:
                    var_to_crit[var].add(v)
            else:
                var_to_crit[var].add(new_index)

    return var_to_crit, var_to_digitizer

def merge_successive_range(var_to_crit):

    var_to_list  = {x:list() for x in var_to_crit.keys()}
    for var, crits in var_to_crit.items():
        l = sorted(list(crits))
        i = 0
        while i <= len(l)-2:
            sub = list()
            while l[i]+1 == l[i+1] and i <= len(l)-2:
                sub.append(l[i])
                i += 1

            sub.append(l[i])
            i += 1
            var_to_list[var].append(sub)

        if l[i]-1 == l[i-1]:
            var_to_list[var][-1].append(l[i])
        else:
            var_to_list[var].append([l[i]])
    return var_to_list

def plot_critical(map_var_val, var_to_crit, var_to_digitizer):

    for var, values in var_to_crit.items():
        digitizer = var_to_digitizer[var]
        zones = set()
        for crit in values:
            lower, upper = digitizer.get_range_extreme(crit)
            zones.add((lower, upper))

        x = np.linspace(0, len(map_var_val[var]), len(map_var_val[var]))
        plt.plot(map_var_val[var], alpha=0.2)
        plt.title(var)

        for z in zones:
            plt.fill_between(x, z[0], z[1], alpha=0.2, color="r")

        plt.show()

def main(conf, output, data, apply_filter):

    actuators, sensors = limitVal.get_actuators_sensors(conf)

    if apply_filter is not None:
        final_data = filter_data(data, actuators)
    else:
        final_data = data

    event_var_critical, event_var_ratio = get_cand_critical_values_from_std(final_data,
                                                                            actuators,
                                                                            sensors)
    filter_based_on_ratio(data, event_var_critical, event_var_ratio)
    var_min_split = get_max_split_per_var(event_var_critical)

    var_to_crit, var_to_digitizer = get_var_to_critical_value(event_var_critical, var_min_split)

    var_to_list = merge_successive_range(var_to_crit)

    with open(conf) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for variable in desc["variables"]:
            var = variable["variable"]
            name = var["name"]
            if name in var_to_list:
                var["critical"] = var_to_list[name]
                var["digitizer"] = var_to_digitizer[name].serialize()

        with open(output, "w") as ofh:
            content = yaml.safe_dump(desc, allow_unicode=False)
            ofh.write(content)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf_file",
                        help="configuration file of variable")

    parser.add_argument("--input", action="store", dest="input",
                        help="binary file of process")

    parser.add_argument("--output", action="store", dest="output",
                        help="output file with the critical values")

    parser.add_argument("--filter", action="store_true", dest="apply_filter",
                        help="apply_filter")

    args = parser.parse_args()

    data = read_state_file(args.input)

    main(args.conf_file, args.output, data, args.apply_filter)

