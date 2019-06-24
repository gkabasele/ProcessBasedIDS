import argparse
from copy import deepcopy
from collections import Iterable
from operator import attrgetter
import math
import pickle
import pdb
from bisect import insort

import yaml
import jsonpickle
import pprint
from utils import *

DIST = 0.005

MAGNITUDE = 0.1

CACHE_ON = "./sensors_on.bin"
CACHE_OFF = "./sensors_off.bin"

class CandVal(object):

    def __init__(self, val, nbr_seen=1, j_measure=0):
        self.nbr_seen = nbr_seen
        self.cur_avg = val
        self.j_measure = j_measure

    def add_value(self, val):
        self.nbr_seen += 1
        self.cur_avg = self.cur_avg + (val - self.cur_avg)/self.nbr_seen

    def __repr__(self):
        return "{} ({})".format(self.cur_avg, self.nbr_seen)

    def __hash__(self):
        return self.cur_avg.__hash__()

    def __lt__(self, other):
        return self.j_measure < other.j_measure

    def __gt__(self, other):
        return self.j_measure > other.j_measure


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

def display_dict(d1, d2):
    for key in d1:
        print("Act:{}".format(key))
        print("On Value")
        for k, v in d1[key].items():
            if len(v) < 10 and len(v) != 0:
                print("{}:{}".format(k, v))
        print("\nOff Value")
        for k, v in d2[key].items():
            if len(v) < 10 and len(v) != 0:
                print("{}:{}".format(k,v))
        print("---------------------\n")

def sort_candidate(d1, d2):
    for key in d1:
        for _, val in d1[key].items():
            val = sorted(val, key=attrgetter('nbr_seen', 'cur_avg'))
        for _, val in d2[key].items():
            val = sorted(val, key=attrgetter('nbr_seen', 'cur_avg'))

def create_sensors_cand(actuators, actuators_trans, state, pv_store):
    sensors = {}

    for k in actuators:
        # Sensor name, readings
        for key, val in actuators[k].items():
            #Only candidate value
            if key not in sensors:
                sensors[key] = {}

            pv = pv_store[key]
            for r in val:
                found = False
                for x in sensors[key].keys(): 
                    if same_value(r, x, pv):
                        found = True
                        break
                if not found and r.nbr_seen >= 0.75*actuators_trans[k][state]:
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
    for i in range(len(data) - 1):
        state = data[i]
        for k in sensors_on:
            pv = pv_store[k]
            for val in sensors_on[k]:
                if same_value(state[k], val, pv):
                    probval = sensors_on[k][val]
                    probval.prob += 1
                    for act in probval:
                        if i > 0:
                            state_prev = data[i-1]
                            if ((state_prev[act] == 0 or state_prev[act] == 1) and
                                    (state[act] == 2)):
                                probval.act_prob[act] += 1

        for k in sensors_off:
            pv = pv_store[k]
            for val in sensors_off[k]:
                if same_value(state[k], val, pv):
                    probval = sensors_off[k][val]
                    probval.prob += 1
                    for act in probval:
                        if i > 0:
                            state_prev = data[i-1]
                            if ((state_prev[act] == 2) and
                                    (state[act] == 1 or state[act] == 0)):
                                probval.act_prob[act] += 1

def normalized_prob(data, actuators_prob, actuators_trans, sensors):
    n = len(data)

    for key in actuators_prob:
        actuators_prob[key]["on"] = actuators_prob[key]["on"]/n
        actuators_prob[key]["off"] = actuators_prob[key]["off"]/n

        actuators_trans[key]["onon"] = actuators_trans[key]["onon"]/(n-1)
        actuators_trans[key]["onoff"] = actuators_trans[key]["onoff"]/(n-1)
        actuators_trans[key]["offon"] = actuators_trans[key]["offon"]/(n-1)
        actuators_trans[key]["offoff"] = actuators_trans[key]["offoff"]/(n-1)

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
                      actuators_prob, actuators_trans, actuator_name,
                      actuator_state): 
    pv = pv_store[sensor_name]
    p_sensor = None
    p_act_cond = None
    p_act = actuators_trans[actuator_name][actuator_state]
    for key, val in sensors[sensor_name].items():
        if same_value(key, sensor_value, pv):
            p_sensor = val.prob
            p_act_cond = val.act_prob[actuator_name]
            break
    return p_sensor * _compute_j_measure(p_act_cond, p_act)

def order_magnitude(val1, val2):

    if val1 >= val2 and val1 != 0:
        return val2/val1
    elif val2 > val1 and val2 != 0:
        return val1/val2
    else:
        return 0

def cand_rule_quality(pv_store, sensors_on, sensors_off,
                      actuators_on, actuators_off,
                      actuators_prob, actuators_trans):
    on_rule = dict(zip(actuators_on.keys(), [{} for _ in range(len(actuators_on))]))
    off_rule = copy.deepcopy(on_rule)
    for key in actuators_on:
        for k, v in actuators_on[key].items():
            if len(v) < 3 and len(v) != 0:
                for value in v:
                    j_measure = compute_j_measure(pv_store, sensors_on, k,
                                                  value, actuators_prob,
                                                  actuators_trans, key, "offon")
                    value.j_measure = j_measure

                    if k not in on_rule[key]:
                        on_rule[key][k] = [value]
                    else:
                        insort(on_rule[key][k], value)

        for k, v in actuators_off[key].items():
            if len(v) < 3 and len(v) != 0:
                for value in v:
                    j_measure = compute_j_measure(pv_store, sensors_off, k,
                                                  value, actuators_prob, actuators_trans,
                                                  key, "onoff")
                    if k not in off_rule[key]:
                        off_rule[key][k] = [value]
                    else:
                        insort(off_rule[key][k], value)

    return on_rule, off_rule

def main(phys, variables, cache_on, cache_off):

    data = pickle.load(open(phys, "rb"))
    pv_store = {}
    setup(variables, pv_store)

    actuators_on = create_actuators(pv_store)
    actuators_off = deepcopy(actuators_on)
    actuators = dict(zip(actuators_on.keys(), [0]* len(actuators_on)))
    actuators_prob = dict(zip(actuators_on.keys(),
                              [{"on": 0, "off":0} for i in range(len(actuators_on))]))
    actuators_trans = dict(zip(actuators_on.keys(),
            [{"onoff": 0, "onon": 0, "offon":0, "offoff":0} for i in range(len(actuators_on))]))

    # Find candidate that impact actuators
    for i, state in enumerate(data):
        if i == 0:
            for k in actuators:
                actuators[k] = state[k]
                if state[k] == 1 or state[k] == 0:
                    actuators_prob[k]["off"] += 1
                elif state[k] == 2:
                    actuators_prob[k]["on"] += 1
        else:
            for k in actuators:
                if state[k] == 1 or state[k] == 0:
                    actuators_prob[k]["off"] += 1

                elif state[k] == 2:
                    actuators_prob[k]["on"] += 1

                if actuators[k] != state[k]:
                    if ((actuators[k] == 1 and state[k] == 0) or
                            (actuators[k] == 0 and state[k] == 1)):
                        actuators_trans[k]["offoff"] += 1

                    if (actuators[k] == 2 and (state[k] == 0 or state[k] == 1)):
                        get_readings(k, state, pv_store, actuators_off)
                        actuators_trans[k]["onoff"] += 1

                    elif ((actuators[k] == 1 or actuators[k] == 0) and state[k] == 2):
                        get_readings(k, state, pv_store, actuators_on)
                        actuators_trans[k]["offon"] += 1

                    actuators[k] = state[k]
                else:
                    if actuators[k] == 1 or actuators[k] == 0:
                        actuators_trans[k]["offoff"] += 1

                    elif actuators[k] == 2:
                        actuators_trans[k]["onon"] += 1

    #sort_candidate(actuators_on, actuators_off)
    sensors_on = None
    sensors_off = None
    # Filter non impacting sensors
    if isinstance(cache_on, bool) and cache_on:
        sensors_on = create_sensors_cand(actuators_on, actuators_trans, "offon", pv_store)
        sensors_off = create_sensors_cand(actuators_off, actuators_trans, "onoff", pv_store)

        compute_frequence(data, sensors_on, sensors_off, pv_store)

        with open(CACHE_ON, "w") as f:
            f.write(jsonpickle.encode(sensors_on))

        with open(CACHE_OFF, "w") as f:
            f.write(jsonpickle.encode(sensors_off))

    elif isinstance(cache_on, str):

        with open(CACHE_ON, "r") as f:
            sensors_on = jsonpickle.decode(f.read())

        with open(CACHE_OFF, "r") as f:
            sensors_off = jsonpickle.decode(f.read())

    pprint.pprint(actuators_prob)
    pprint.pprint(actuators_trans)
    pdb.set_trace()

    normalized_prob(data, actuators_prob, actuators_trans, [sensors_on, sensors_off])
    #j_measure_lit101 = compute_j_measure(pv_store, sensors_on, "lit101",
    #                                     495.0656, actuators_prob, actuators_trans,
    #                                     "mv101", "onoff")
    #on_rule, off_rule = cand_rule_quality(pv_store, sensors_on, sensors_off,
    #                                      actuators_on, actuators_off,
    #                                      actuators_prob)

    #print(j_measure_lit101)
    #print("On Value")
    #pprint.pprint(on_rule)

    #print("Off Value")
    #pprint.pprint(off_rule)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--phys", action="store", dest="phys")
    parser.add_argument("--vars", action="store", dest="vars")
    parser.add_argument("--cache-on", action="store_const", dest="cache_on",
                        const=True, default=CACHE_ON)
    parser.add_argument("--cache-off", action="store_const", dest="cache_off",
                        const=True, default=CACHE_OFF)
    args = parser.parse_args()

    main(args.phys, args.vars, args.cache_on, args.cache_off)
