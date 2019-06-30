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

DIST = 0.01

MAGNITUDE = 0.1

TRESH = 0.0001

CACHE_ON = "./sensors_on.bin"
CACHE_OFF = "./sensors_off.bin"

class CandVal(object):

    def __init__(self, val=0, nbr_seen=1, j_measure=0):
        self.nbr_seen = nbr_seen
        self.cur_avg = val
        self.j_measure = j_measure
        self.prob = 0

    def add_value(self, val):
        self.nbr_seen += 1
        self.cur_avg = self.cur_avg + (val - self.cur_avg)/self.nbr_seen

    def normalize(self, nbr):
        if nbr != 0 and nbr >= self.nbr_seen:
            self.prob = self.nbr_seen/nbr

    def __repr__(self):
        return "{} ({})".format(self.cur_avg, self.nbr_seen)

    def __hash__(self):
        return self.cur_avg.__hash__()
    
    def __lt__(self, other):
        return self.j_measure < other.j_measure

    def __gt__(self, other):
        return self.j_measure > other.j_measure

    def from_json(self, text):
        val, freq = text.split()
        self.cur_avg = float(val)
        self.nbr_seen = int(freq.replace('(', '').replace(')', ''))
        return self

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

    def __getitem__(self, key):
        return self.readings.__getitem__(key)

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

def export_critical(filename, output, critical_vals):
    with open(filename) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for sens, crit_vals in critical_vals.items():
            val = [x.cur_avg for x in crit_vals]
            for variable in desc['variables']:
                var = variable['variable']
                if var['name'] == sens:
                    var["critical"] = val
                    break
        with open(output, "w") as ofh:
            content = yaml.dump(desc)
            ofh.write(content)

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

def normalize_readings(actuators, actuators_trans, state):
    for k in actuators:
        for key, readings in actuators[k].items():
            for candval in readings:
                candval.normalize(actuators_trans[k][state])

def create_sensors_cand(actuators, actuators_trans, state, pv_store):
    sensors = {}

    for k in actuators:
        # Sensor name, readings
        for key, readings in actuators[k].items():
            #Only candidate value
            if key not in sensors:
                sensors[key] = {}

            pv = pv_store[key]
            #pdb.set_trace()

            for candval in readings:
                candval.normalize(actuators_trans[k][state])
                # in 75% of the case when there was a switch, we observed this value
                if candval not in sensors[key] and candval.prob >= 0.75:
                    probval = ProbVal()
                    #probval.add_act_prob(actuators.keys())
                    sensors[key][candval] = probval
    return sensors

def decode_sensors_cand(sensors_enc):
    for k in sensors_enc:
        candvals = [CandVal().from_json(x) for x in sensors_enc[k].keys()]
        probvals = sensors_enc[k].values()
        sensors_enc[k] = dict(zip(candvals, probvals))


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
                    #for act in probval:
                    #    if i > 0:
                    #        state_prev = data[i-1]
                    #        if ((state_prev[act] == 0 or state_prev[act] == 1) and
                    #                (state[act] == 2)):
                    #            probval.act_prob[act] += 1

        for k in sensors_off:
            pv = pv_store[k]
            for val in sensors_off[k]:
                if same_value(state[k], val, pv):
                    probval = sensors_off[k][val]
                    probval.prob += 1
                    #for act in probval:
                    #    if i > 0:
                    #        state_prev = data[i-1]
                    #        if ((state_prev[act] == 2) and
                    #                (state[act] == 1 or state[act] == 0)):
                    #            probval.act_prob[act] += 1

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
                #probval.normalized_act()
                probval.prob = probval.prob/n

def _compute_j_measure(p_cond, p_x):
    try:
        res = p_cond * math.log2((p_cond/p_x)) + (1 - p_cond) * math.log2((1-p_cond)/(1-p_x))
    except ValueError:
        if p_cond == 1:
            res =  math.log2((p_cond/p_x))
        elif p_cond == 0:
            res = math.log2(1/(1-p_x))
        
    return res

def compute_j_measure(pv_store, sensors, sensor_name, sensor_value,
                      actuators, actuators_prob, actuators_trans,
                      actuator_name, actuator_state):

    pv = pv_store[sensor_name]
    p_sensor = 0
    p_sens_given_act = 0
    p_act = actuators_trans[actuator_name][actuator_state]
    if sensor_name in actuators[actuator_name]:
        for candval in actuators[actuator_name][sensor_name]:
            if same_value(candval, sensor_value, pv):
                p_sens_given_act = candval.prob
                break

    for key, probval in sensors[sensor_name].items():
        if same_value(key, sensor_value, pv):
            p_sensor = probval.prob
            break

    return p_act * _compute_j_measure(p_sens_given_act, p_sensor)

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
    on_rule = dict(zip(actuators_on.keys(), [[] for _ in range(len(actuators_on))]))
    off_rule = copy.deepcopy(on_rule)
    for key in actuators_on:
        for k, v in actuators_on[key].items():
            for value in v:
                if value.prob >= 0.75:
                    j_measure = compute_j_measure(pv_store, sensors_on, k,
                                              value, actuators_on, actuators_prob,
                                              actuators_trans, key, "offon")
                    value.j_measure = j_measure
                    if j_measure >= TRESH:
                        on_rule[key].append((k, value))

        for k, v in actuators_off[key].items():
            for value in v:
                if value.prob >= 0.75:
                    j_measure = compute_j_measure(pv_store, sensors_off, k,
                                                  value, actuators_off, actuators_prob,
                                                  actuators_trans, key, "onoff")
                    value.j_measure = j_measure
                    if j_measure >= TRESH:
                        off_rule[key].append((k, value))

    return on_rule, off_rule

def summarize_critical_val(critical_vals, pv_store):
    summary = dict(zip(critical_vals.keys(), [[] for _ in range(len(critical_vals))]))
    for k, vals in critical_vals.items():
        i = 0
        pv = pv_store[k]
        while i<len(vals) - 1:
            summary[k].append(vals[i])
            j = i + 1
            while j < len(vals):
                if same_value(vals[i], vals[j], pv):
                    i += 1
                    j += 1
                else:
                    i+= 1
                    break
    return summary        

def get_critical_val(on_rule, off_rule, pv_store):
    critical_vals = {}

    for k in on_rule:
        for reading in on_rule[k]:
            sens, val = reading
            if sens not in critical_vals:
                critical_vals[sens] = [val]
            else:
                critical_vals[sens].append(val)

        for reading in off_rule[k]:
            sens, val = reading
            if sens not in critical_vals:
                critical_vals[sens] = [val]
            else:
                critical_vals[sens].append(val)

    for k, val in critical_vals.items():
        val.sort(key=lambda x: x.cur_avg)

    return summarize_critical_val(critical_vals, pv_store)

def main(phys, variables, output, cache_on, cache_off):

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

        with open(cache_on, "w") as f:
            f.write(jsonpickle.encode(sensors_on))

        with open(cache_off, "w") as f:
            f.write(jsonpickle.encode(sensors_off))

    elif isinstance(cache_on, str):

        with open(cache_on, "r") as f:
            sensors_on = jsonpickle.decode(f.read())

        with open(cache_off, "r") as f:
            sensors_off = jsonpickle.decode(f.read())
        normalize_readings(actuators_on, actuators_trans, "offon")
        normalize_readings(actuators_off, actuators_trans, "onoff")
        decode_sensors_cand(sensors_on)
        decode_sensors_cand(sensors_off)

    normalized_prob(data, actuators_prob, actuators_trans, [sensors_on, sensors_off])
    pprint.pprint(actuators_prob)
    pprint.pprint(actuators_trans)
    #j_measure_lit101 = compute_j_measure(pv_store, sensors_on, "lit101", 495.0656,
    #                                     actuators_on, actuators_prob, actuators_trans,
    #                                     "mv101", "offon")
    #j_measure_lit101 = compute_j_measure(pv_store, sensors_on, "lit101",
    #                                     495.0656, actuators_prob, actuators_trans,
    #                                     "mv101", "onoff")
    on_rule, off_rule = cand_rule_quality(pv_store, sensors_on, sensors_off,
                                          actuators_on, actuators_off,
                                          actuators_prob, actuators_trans)

    #print(j_measure_lit101)
    critical_vals = get_critical_val(on_rule, off_rule, pv_store)
    export_critical(variables, output, critical_vals)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--phys", action="store", dest="phys")
    parser.add_argument("--vars", action="store", dest="vars")
    parser.add_argument("--output", action="store", dest="output")
    parser.add_argument("--cache-on", action="store_const", dest="cache_on",
                        const=True, default=CACHE_ON)
    parser.add_argument("--cache-off", action="store_const", dest="cache_off",
                        const=True, default=CACHE_OFF)
    args = parser.parse_args()

    main(args.phys, args.vars, args.output, args.cache_on, args.cache_off)
