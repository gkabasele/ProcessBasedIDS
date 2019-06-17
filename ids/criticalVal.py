import argparse
from copy import deepcopy
from collections import Iterable
import pickle
import pdb

import yaml
import pprint
from utils import *

DIST = 0.01

class Readings(object):

    def __init__(self):
        self.readings = []
        self.times = 0

    def add_reading(self, pv, new, debug=False):
        #if debug:
        #        pdb.set_trace()

        if len(self.readings) == 0:
            self.readings.append(new)
        else:
            test = [pv.normalized_dist(new, val) <= DIST for val in self.readings]
            if not any(test):
                self.readings.append(new)
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

    def __str__(self):
        return "Prob: {}, Act: {}".format(self.prob, self.act_prob)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return self.act_prob.__iter__()

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
            if len(v) < 3 and len(v) != 0 :
                print("{}:{}".format(k, v))
        print("\nOff Value")
        for k, v in e[key].items():
            if len(v) < 3 and len(v) != 0 :
                print("{}:{}".format(k,v))
        print("---------------------\n")

def create_sensors_cand(actuators):
    sensors = {} 

    for k in actuators:
        for key, val in actuators[k].items():
            if key not in sensors:
                sensors[key] = {}

            for r in val:
                if r not in sensors[key]:
                    probval = ProbVal()
                    probval.add_act_prob(actuators.keys())
                    sensors[key][r] = probval
    return sensors


def get_readings(name, state, pv_store, reading, debug=False):

    try:
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

    except KeyError as err:
        pass

def compute_frequence(data, sensors):
    for state in data:
        for k in sensors:
            for val in sensors[k]:
                if state[k] == val:
                    probval = sensors[k][val]
                    probval.prob += 1
                    for act in probval:
                        if state[act] == 2:
                            probval.act_prob[act] += 1

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
                try:
                    actuators[k] = state[k]
                except KeyError as err:
                    pass
        else:
            for k in actuators:
                try:
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
                except KeyError as err:
                    pass

    # Filter non impacting sensors
    sensors_on = create_sensors_cand(actuators_on)
     
    pprint.pprint(sensors_on)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--phys", action="store", dest="phys")
    parser.add_argument("--vars", action="store", dest="vars")
    args = parser.parse_args()

    main(args.phys, args.vars)
