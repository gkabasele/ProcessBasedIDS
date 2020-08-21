import pdb
import math
import argparse
from collections import Counter

import yaml
import pprint
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from utils import *
import predicate as pd
import limitVal
from timeSeriesAnalysis import Digitizer

"""
J-Measure approach. Looking for values occuring when one of the
actuators is changing. Taking into account: probability of actuators changing,
probability of sensors values and conditional probability
"""

DIST = 0.01

MAGNITUDE = 0.1

TRESH = 0.0001

class EventCriticalVal(object):

    def __init__(self, event_name):
        self.event_name = event_name
        self.var_to_val = dict()

    def __str__(self):
        return str(self.var_to_val)

    def __repr__(self):
        return str(self)

def get_all_values(data):
    map_var_val = {x: list() for x in data[0].keys()}

    for state in data:
        for k, v in state.items():
            map_var_val[k].append(state[k])

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

def discretize_data(map_var_val, actuators):
    map_var_discrete = dict()
    for k, v in map_var_val.items():
        if k not in actuators and k not in limitVal.IGNORE_COL:
            min_val = np.min(map_var_val[k])
            max_val = np.max(map_var_val[k])
            d = Digitizer(min_val, max_val)
            map_var_discrete[k] = (d.digitize(v), d)
        else:
            map_var_discrete[k] = map_var_val[k]
    return map_var_discrete

# var->event->Counter
def discrete_event_data(map_var_event_val, map_var_discrete):
    for var, events in map_var_event_val.items():
        for k, values in events.items():
            events[k] = Counter(map_var_discrete[var][1].digitize(values))

def get_prob_from_hist(values, value):
    ar, bin_edges = np.histogram(values, bins=10)
    i = np.digitize(value, bin_edges) - 1
    # if value beyond bound, 0 or len(bins) is returned
    # len(bin) == len(ar) + 1
    if i == len(ar):
        i = len(ar)-1
    elif i < 0:
        i = 0
    return ar[i]/ar.sum()

def get_event_prob(data, event):
    nb_trans_occur = actuator_nbr_occurence(data[event.varname],
                                            event.from_value)
    return len(event.timestamps)/(nb_trans_occur-1)

#Find probability of observing a value in a timeseries
def get_value_prob(var, map_var_values, value):
    return get_prob_from_hist(map_var_values[var][0], value)

# Get the number of time a actuator has a particular
# state
# FIXME testing
def actuator_nbr_occurence(values, act_state):
    c = Counter(values)
    pdb.set_trace()
    if act_state == pd.ON:
        return c[pd.ON]
    else:
        return c[pd.OFF] + c[pd.OFFZ]

# Get the probability of observing a value given the
# occurence of an event
def get_value_prob_given_event(map_var_event_val, event, var, value):
    return map_var_event_val[var][event][value]/sum(map_var_event_val[var][event].values())

#map_var_values: var->[val0, val1, ...]
#map_var_event_val: var->(eventa->[vala0, vala1, ...], eventb->[valb0, valb1,...],..)
def compute_j_measure(map_var_values, map_var_event_val, event, var, value):
    prob_event = get_event_prob(map_var_values, event)
    prob_value = get_value_prob(var, map_var_values, value)
    prob_value_given_event = get_value_prob_given_event(map_var_event_val, event, var, value)

    if "mv101" in str(event):
        pdb.set_trace()

    # website
    #j_measure = ((prob_event * prob_value_given_event * math.log(prob_value_given_event/prob_value)) +
    #             prob_event * (1-prob_value_given_event) * math.log((1-prob_value_given_event)/(1-prob_value)))
    # paper
    if prob_value_given_event == 0:
        j_measure = math.log(1/(1-prob_value))

    elif prob_value_given_event == 1:
        j_measure = prob_event * math.log(1/prob_value)

    else:
        j_measure = ((prob_event * prob_value_given_event * math.log(prob_value_given_event/prob_value)) +
                     (1-prob_value_given_event) * math.log((1-prob_value_given_event)/(1-prob_value)))

    return j_measure

def compute_rule_support(map_var_event_val, event, var, value):
    return get_value_prob_given_event(map_var_event_val, event, var, value)

def get_candidate_value_from_j_measure(map_var_values, map_var_event_val):
    var = "lit101"
    for event, values in map_var_event_val[var].items():
        for value in values:
            print("Starting value:{} for event {}".format(value, event))
            j = compute_j_measure(map_var_values, map_var_event_val, event, var, value)

def get_candidate_value_from_support(map_var_event_val):
    critical_values = {x: EventCriticalVal(x.varname) for x in map_var_event_val[list(map_var_event_val.keys())[0]]}
    for var in map_var_event_val:
        print("Starting variable:{}".format(var))
        for event, values in map_var_event_val[var].items():
            for value in values:
                support = compute_rule_support(map_var_event_val, event, var, value)

                if support >= 0.9:
                    critical_values[event].var_to_val[var] = value

    pdb.set_trace() 

def main(conf, data, apply_filter):
    actuators = limitVal.get_actuators(conf)

    if apply_filter is not None:
        final_data = filter_data(data, actuators)

    else:
        final_data = data

    var_to_list = get_all_values(final_data)
    discrete_var_to_list = discretize_data(var_to_list, actuators)

    events = pd.retrieve_update_timestamps(actuators, final_data)
    map_var_event_val = {x:None for x in final_data[0].keys() if x not in actuators and x not in limitVal.IGNORE_COL}
    limitVal.get_values_and_event_values(final_data, actuators, map_var_event_val, events)

    discrete_event_data(map_var_event_val, discrete_var_to_list)
    #get_candidate_value_from_j_measure(discrete_var_to_list, map_var_event_val)
    get_candidate_value_from_support(map_var_event_val)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf_file",
                        help="configuration file of variable")
    parser.add_argument("--input", action="store", dest="input",
                        help="binary file of process")
    parser.add_argument("--filter", action="store_true", dest="apply_filter",
                        help="apply_filter")

    args = parser.parse_args()

    data = read_state_file(args.input)

    main(args.conf_file, data, args.apply_filter)

