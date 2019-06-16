import argparse
import yaml
import pickle
import pdb
from utils import *
from copy import deepcopy


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

def display_dict(d):
    for key, value in d.items():
        print("Act:{}".format(key))
        for k, v in d[key].items():
            if len(v) < 10:
                print("{}:{}".format(k, v))
        print("---------------------\n")

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

def main(phys, variables):

    data = pickle.load(open(phys, "rb"))    
    pv_store = {}
    setup(variables, pv_store)

    actuators_on = create_actuators(pv_store)
    actuators_off = deepcopy(actuators_on)
    actuators = dict(zip(actuators_on.keys(), [0]* len(actuators_on)))

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
                    if actuators[k] != state[k]:
                        actuators[k] = state[k]

                        if actuators[k] == 1:
                            get_readings(k, state, pv_store, actuators_off)

                        elif actuators[k] == 2:
                            get_readings(k, state, pv_store, actuators_on,
                                         k == "mv101")
                except KeyError as err:
                    pass

    display_dict(actuators_on)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--phys", action="store", dest="phys")
    parser.add_argument("--vars", action="store", dest="vars")
    args = parser.parse_args()

    main(args.phys, args.vars)
