import argparse
from copy import deepcopy
from collections import Iterable
from operator import itemgetter, attrgetter
import math
import pickle
import pdb

import yaml
import pprint
from utils import *

DIST = 0.01


class CandVal(object):

    def __init__(self, val):
        self.nbr_seen = 1
        self.cur_avg = val

    def add_value(self, val):
        self.nbr_seen += 1
        self.cur_avg = self.cur_avg + (val - self.cur_avg)/self.nbr_seen

    def __repr__(self):
        return "{} ({})".format(self.cur_avg, self.nbr_seen)

    def __hash__(self):
        return self.cur_avg.__hash__()

class Readings(object):

    def __init__(self):
        self.readings = []
        self.times = 0

    def add_reading(self, pv, new, debug=False):
        #if debug:
        #        pdb.set_trace()

        if len(self.readings) == 0:
            val = CandVal(new)
            self.readings.append(val)
        else:
            found = False
            for val in self.readings:
                if pv.normalized_dist(new, val.cur_avg) <= DIST:
                    val.add_value(new)
                    found = True
                    break
            if not found:
                val = CandVal(new)
                self.readings.append(val)
        self.times += 1


    def __len__(self):
        return len(self.readings)

    def __str__(self):
        return str(self.readings)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return self.readings.__iter__()

class ProbVal(object):

    def __init__(self):
        self.prob = 0
        self.act_prob = {}

    def add_act_prob(self, key):
        if isinstance(key, str):
            self.act_prob[key] = 0.0
        elif isinstance(key, Iterable):
            for k in key:
                self.act_prob[k] = 0.0

    def normalized_act(self):
        for act in self.act_prob:
            self.act_prob[act] = self.act_prob[act]/self.prob

    def __str__(self):
        return "Prob: {}, Act: {}".format(self.prob, self.act_prob)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return self.act_prob.__iter__()


def same_value(val1, val2, pv):
    if isinstance(val1, CandVal) and isinstance(val2, CandVal):
        return pv.normalized_dist(val1.cur_avg, val2.cur_avg) <= DIST

    elif isinstance(val1, CandVal):
        return pv.normalized_dist(val1.cur_avg, val2) <= DIST

    elif isinstance(val2, CandVal):
        return pv.normalized_dist(val1, val2.cur_avg) <= DIST

def get_bool_vars(pv_store):
    return [x for x in pv_store if pv_store[x].is_bool_var()]

def get_num_vars(pv_store):
    return [x for x in pv_store if not pv_store[x].is_bool_var()]

def setup(filename, pv_store):
    with open(filename) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for var_desc in desc['variables']:
            var = var_desc['variable']
            if var['type'] == DIS_COIL or var['type'] == DIS_INP:
                limit_values = [1, 2]
                min_val = 1
                max_val = 2
            else:
                limit_values = var['values']
                min_val = var['min']
                max_val = var['max']

            pv = ProcessSWaTVar(var['name'], var['type'],
                                limit_values=limit_values,
                                min_val=min_val,
                                max_val=max_val)
            pv_store[pv.name] = pv

def create_actuators(pv_store):
    actuators_name = get_bool_vars(pv_store) 
    sensor_reading = get_num_vars(pv_store)
    actuators = {}

    for name in actuators_name:
        readings = [Readings() for i in range(len(sensor_reading))]
        actuators[name] = dict(zip(sensor_reading, readings))

    return actuators

def display_dict(d, e):
    for key in d:
        print("Act:{}".format(key))
        print("On Value")
        for k, v in d[key].items():
            if len(v) < 3 and len(v) != 0:
                print("{}:{}".format(k, v))
        print("\nOff Value")
        for k, v in e[key].items():
            if len(v) < 3 and len(v) != 0:
                print("{}:{}".format(k,v))
        print("---------------------\n")

def sort_candidate(d, e):
    for key in d:
        for k, val in d[key].items():
            val = sorted(val, key=attrgetter('nbr_seen', 'cur_avg'))
        for k, val in e[key].items():
            val = sorted(val, key=attrgetter('nbr_seen', 'cur_avg'))

def create_sensors_cand(actuators, pv_store):
    sensors = {}

    for k in actuators:
        # Sensor name, readings
        for key, val in actuators[k].items():
            #Only candidate value
            if len(val) < 3 and len(val) != 0:
                if key not in sensors:
                    sensors[key] = {}

                pv = pv_store[key]
                for r in val:
                    if not any([same_value(r, x, pv) for x in sensors[key].keys()]):
                        probval = ProbVal()
                        probval.add_act_prob(actuators.keys())
                        sensors[key][r] = probval
    return sensors


def get_readings(name, state, pv_store, reading, debug=False):

    """
    if debug:
        print("Sensors:{}".format(name))
        print(state)
        print("TS: {}".format(state['timestamp']))
    """
    for k in get_num_vars(pv_store):
        pv = pv_store[k]
        val = state[k]
        #if debug:
        #    print("Actuators:{}".format(k))
        reading[name][k].add_reading(pv, val, (debug and k == "lit101"))


def compute_frequence(data, sensors_on, sensors_off, pv_store):
    for state in data:
        for k in sensors_on:
            pv = pv_store[k]
            for val in sensors_on[k]:
                if same_value(state[k], val, pv):
                    probval = sensors_on[k][val]
                    probval.prob += 1
                    for act in probval:
                        if state[act] == 2:
                            probval.act_prob[act] += 1

        for k in sensors_off:
            pv = pv_store[k]
            for val in sensors_off[k]:
                if same_value(state[k], val, pv):
                    probval = sensors_off[k][val]
                    probval.prob += 1
                    for act in probval:
                        if state[act] == 1 or state[act] == 0:
                            probval.act_prob[act] += 1

def normalized_prob(data, actuators, sensors):
    n = len(data)

    for act, state in actuators.items():
        state["on"] = state["on"]/n
        state["off"] = state["off"]/n

    for sensor in sensors:
        for key in sensor:
            for val, probval in sensor[key].items():
                probval.normalized_act()
                probval.prob = probval.prob/n

def _compute_j_measure(p_cond, p_x):
    try:
        res = p_cond * math.log((p_cond/p_x), 2) + (1 - p_cond) * math.log((1-p_cond)/(1-p_x), 2)
    except ValueError:
        res = 0
    return res

def compute_j_measure(pv_store, sensors, sensor_name, sensor_value,
                      actuators_prob, actuator_name, actuator_state): 
    pv = pv_store[sensor_name]
    p_sensor = None
    p_act_cond = None
    p_act = actuators_prob[actuator_name][actuator_state]
    for key, val in sensors[sensor_name].items():
        if same_value(key, sensor_value, pv):
            p_sensor = val.prob
            p_act_cond = val.act_prob[actuator_name]
            break
    return p_sensor * _compute_j_measure(p_act_cond, p_act)

def cand_rule_quality(pv_store, sensors_on, sensors_off,
                      actuators_on, actuators_off,
                      actuators_prob, thresh):
    on_rule = dict(zip(actuators_on.keys(), [{} for _ in range(len(actuators_on))]))
    off_rule = copy.deepcopy(on_rule)
    for key in actuators_on:
        for k, v in actuators_on[key].items():
            if len(v) < 3 and len(v) != 0:
                for value in v:
                    j_measure = compute_j_measure(pv_store, sensors_on, k,
                                                  value, actuators_prob, key, "on")
                    if j_measure >= thresh:
                        if k not in on_rule[key]:
                            on_rule[key][k] = [value]
                        else:
                            on_rule[key][k].append(value)

        for k, v in actuators_off[key].items():
            if len(v) < 3 and len(v) != 0:
                for value in v:
                    j_measure = compute_j_measure(pv_store, sensors_off, k,
                                                  value, actuators_prob, key, "off")
                    if j_measure >= thresh:
                        if k not in off_rule[key]:
                            off_rule[key][k] = [value]
                        else:
                            off_rule[key][k].append(value)

    return on_rule, off_rule

def main(phys, variables):

    data = pickle.load(open(phys, "rb"))
    pv_store = {}
    setup(variables, pv_store)

    actuators_on = create_actuators(pv_store)
    actuators_off = deepcopy(actuators_on)
    actuators = dict(zip(actuators_on.keys(), [0]* len(actuators_on)))
    actuators_prob =  dict(zip(actuators_on.keys(),
                               [{"on": 0, "off":0} for i in range(len(actuators_on))]))

    # Find candidate that impact actuators
    for i, state in enumerate(data):
        if i == 0:
            for k in actuators:
                actuators[k] = state[k]
        else:
            for k in actuators:
                if state[k] == 1 or state[k] == 0:
                    actuators_prob[k]["off"] += 1

                elif state[k] == 2:
                    actuators_prob[k]["on"] += 1

                if actuators[k] != state[k]:
                    if ((actuators[k] == 1 and state[k] == 0) or
                            (actuators[k] == 0 and state[k] == 1)):
                        actuators[k] = state[k]
                        continue

                    actuators[k] = state[k]

                    if actuators[k] == 1 or actuators[k] == 0:
                        get_readings(k, state, pv_store, actuators_off)

                    elif actuators[k] == 2:
                        get_readings(k, state, pv_store, actuators_on,
                                     k == "mv101")

    # Filter non impacting sensors
    sensors_on = create_sensors_cand(actuators_on, pv_store)
    sensors_off = create_sensors_cand(actuators_off, pv_store)

    compute_frequence(data, sensors_on, sensors_off, pv_store)
    normalized_prob(data, actuators_prob, [sensors_on, sensors_off])
    j_measure_lit101 = compute_j_measure(pv_store, sensors_off, "lit101",
                                         800.9918, actuators_prob, "mv101", "off")
    #on_rule, off_rule = cand_rule_quality(pv_store, sensors_on, sensors_off,
    #                                      actuators_on, actuators_off,
    #                                      actuators_prob, 0.01)

    #print("On Value")
    #pprint.pprint(on_rule)

    #print("Off Value")
    #pprint.pprint(off_rule)
    pprint.pprint(actuators_prob)
    display_dict(actuators_on, actuators_off)
    print(sensors_off["lit101"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--phys", action="store", dest="phys")
    parser.add_argument("--vars", action="store", dest="vars")
    args = parser.parse_args()

    main(args.phys, args.vars)
