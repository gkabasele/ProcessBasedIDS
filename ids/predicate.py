import argparse
import operator
import pdb
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from linearRegression import PredicateLinearRegression
from pvStore import PVStore
from utils import TS
from utils import read_state_file

ON = 2
OFF = 1

TURNON = "turnOn"
TURNOFF = "turnOff"
GT = "greater"
LS = "lesser"

class Predicate(object):
    ops = {
        "<" : operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">" : operator.gt
        }
    pred_id = 0
    def __init__(self, varname, operator, value):

        self.id = Predicate.pred_id
        self.varname = varname
        try:
            self.operator = Predicate.ops[operator]
            self.op_label = operator
        except KeyError:
            raise KeyError("Unknown operator for a Predicate")

        self.support = 0

        self.value = value

        Predicate.pred_id += 1

    def is_true(self, current_value):
        return self.operator(current_value, self.value)

    def __str__(self):
        return "[({}) {} {} {}]".format(self.id, self.varname, self.op_label,
                                        self.value)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash("{} {} {}".format(self.varname, self.op_label, self.value))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        if self.varname != other.varname:
            raise ValueError("Cannot compare two predicates of different variable")

        if self.op_label != other.op_label:
            return self.op_label < other.op_label

        return self.value < other.value

    def __gt__(self, other):
        if self.varname != other.varname:
            raise ValueError("Cannot compare two predicates of different variable")

        if self.op_label != other.op_label:
            return self.op_label > other.op_label

        return self.value > other.value

class Event(object):

    def __init__(self, varname, from_value, to_value):

        self.varname = varname
        self.from_value = from_value
        self.to_value = to_value
        # When did that event occurred
        self.timestamps = []

    def add(self, ts):
        self.timestamps.append(ts)

    def __str__(self):
        return "[{} {}->{}]".format(self.varname, self.from_value, self.to_value)

    def __repr__(self):
        return self.__str__()

def generate_actuators_predicates(actuators, predicates):

    for var in actuators:
        predicates[var] = {ON : [Predicate(var, "==", ON)],
                           OFF: [Predicate(var, "==", OFF)]}

def generate_sensors_predicates(sensors, events, states, 
                                predicates, error_thresh=0.005):

    for act, act_state in events.items():
        for update, event in act_state.items():
            related_sensors = set()
            predicate_for_sensors(related_sensors, sensors, event, states, predicates, error_thresh)

def predicate_for_sensors(related_sensors, sensors, event, states, predicates, error_thresh):

    for sens in sensors:
        if sens not in related_sensors:
            model, X = fit_regression_model(related_sensors, sens, sensors,
                                            event, states)
            predicate_from_model(model, X, sens, event, states, predicates,
                                 related_sensors, error_thresh)

def fit_regression_model(related_sensors, sens, sensors, event, states):
    # get value of all other sensors which are considered as feature
    features = []
    sens_values = []
    for i in event.timestamps:
        state = states[i]
        curr_value = []
        for other_sens in sensors:
            if sens != other_sens:
                curr_value.append(state[other_sens])
            else:
                sens_values.append(state[other_sens])
        features.append(curr_value)
    X = np.array(features)
    Y = np.array(sens_values)
    #model = PredicateLinearRegression(X, Y)
    #model.fit()
    model = LinearRegression()
    model.fit(features, sens_values)
    return model, X

def predicate_from_model(model, X, sens, event, states, predicates, 
                         related_sensors, error_thresh):
    for i, ts in enumerate(event.timestamps):
        sens_value = states[ts][sens]
        prediction = model.predict(X[i].reshape(1, -1))[0]
        error = math.fabs(prediction - sens_value)
        if error <= error_thresh:
            pred_gt = Predicate(sens, ">", prediction + error_thresh)
            pred_lt = Predicate(sens, "<", prediction - error_thresh)
            if sens not in predicates:
                predicates[sens] = {GT : set(), LS: set()}
            predicates[sens][GT].add(pred_gt)
            predicates[sens][LS].add(pred_lt)
        related_sensors.add(sens)

def is_turn_on_event(from_value, to_value):
    return from_value == OFF and to_value == ON

def add_event(i, from_value, to_value, var, events):
    if is_turn_on_event(from_value, to_value):
        if TURNON not in events[var]:
            events[var][TURNON] = Event(var, OFF, ON)

        events[var][TURNON].add(i-1)
    else:
        if TURNOFF not in events[var]:
            events[var][TURNOFF] = Event(var, ON, OFF)

        events[var][TURNOFF].add(i-1)

def retrieve_update_timestamps(actuators, states):

    events = {}
    values = {var: None for var in actuators}

    for i, state in enumerate(states):
        for var in actuators:
            curr_value = state[var]
            if values[var] is None:
                values[var] = curr_value

            elif curr_value != values[var]:
                if var not in events:
                    events[var] = {}

                add_event(i, values[var], curr_value, var, events)
                values[var] = curr_value

    return events

def generate_all_predicates(conf, data):
    store = PVStore(conf)

    actuators = store.discrete_vars()
    sensors = store.continous_vars()

    predicates = {}
    generate_actuators_predicates(actuators, predicates)

    events = retrieve_update_timestamps(store.discrete_vars(), data)

    generate_sensors_predicates(sensors, events, data, predicates, error_thresh=0)

    for sens in sensors:
        predicates[sens][GT] = sorted(list(predicates[sens][GT]), reverse=True)
        predicates[sens][LS] = sorted(list(predicates[sens][LS]))

    return predicates


def main(conf, infile):
    data = read_state_file(infile)
    predicates = generate_all_predicates(conf, data)    
    pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf")
    parser.add_argument("--infile", action="store", dest="infile")

    args = parser.parse_args()

    main(args.conf, args.infile)
