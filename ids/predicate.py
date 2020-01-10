import argparse
import operator
import pdb
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from pvStore import PVStore
from utils import TS
from utils import read_state_file

ON = 2
OFF = 1
OFFZ = 0

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
    pred_id = 1
    def __init__(self, varname, operator, bool_value=None, num_value=None,
                 model=None, error=0):

        self.id = Predicate.pred_id
        self.varname = varname
        try:
            self.operator = Predicate.ops[operator]
            self.op_label = operator
        except KeyError:
            raise KeyError("Unknown operator for a Predicate")

        self.support = 0

        if bool_value is not None:
            self.value = bool_value
            self.model = model
        elif model is not None and num_value is not None:
            self.model = model
            self.value = num_value
            self.error = error
        else:
            raise ValueError("No model or value for the predicate")

        Predicate.pred_id += 1

    def is_true_value(self, current_value):
        return self.operator(current_value, self.value)

    def is_true_model(self, current_value, features): 
        value = self.model.predict(features)[0]
        if self.operator == operator.lt:
            return self.operator(current_value, value - self.error)
        elif self.operator == operator.gt:
            return self.operator(current_value, value + self.error)

    def __str__(self):
        if self.value is not None:
            return "(({}) {} {} {})".format(self.id, self.varname, self.op_label,
                                            self.value)
        elif self.model is not None:
            return "(({}) {} {} {})".format(self.id, self.varname, self.op_label, self.value)

    def __repr__(self):
        return self.__str__()

    #def __hash__(self):
    #    return hash("{} {} {}".format(self.varname, self.op_label, self.value))

    #def __eq__(self, other):
    #    return self.__hash__() == other.__hash__()

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

def generate_actuators_predicates(actuators, store, predicates):

    for var in actuators:
        if not store[var].ignore:
            predicates[var] = {ON : [Predicate(var, "==", ON)],
                               OFF: [Predicate(var, "==", OFF)]}

def generate_sensors_predicates(sensors, store, events, states, 
                                predicates):

    for act, act_state in events.items():
        for update, event in act_state.items():
            related_sensors = set()
            predicate_for_sensors(related_sensors, sensors, store, event, states, predicates)

def predicate_for_sensors(related_sensors, sensors, store, event, states, predicates):

    for sens in sensors:
        if not store[sens].ignore:
            if sens not in related_sensors:
                model, X = fit_regression_model(related_sensors, sens, sensors,
                                                event, states)
                if model is not None:
                    predicate_from_model(store, model, X, sens, sensors, event, states, predicates,
                                         related_sensors)

def fit_regression_model(related_sensors, sens, sensors, event, states):
    # get value of all other sensors which are considered as feature
    other_sens = [x for x in sensors if x != sens]
    features = []
    sens_values = []
    for i in event.timestamps:
        state = states[i]
        curr_value = []
        for other in other_sens:
            curr_value.append(state[other])
        sens_values.append(state[sens])
        features.append(curr_value)
    X = np.array(features)
    Y = np.array(sens_values)
    # Not enough sample for cross validation
    if len(Y) > 4:
        model = LassoCV(normalize=True)
        model.fit(features, sens_values)
        return model, X

    # The event did not occur enough to generate the predicates
    elif len(Y) > 1:
        model = Lasso(alpha=0.1, tol=0.1, max_iter=100000, normalize=True)
        model.fit(features, sens_values)
        return model, X
    else:
        return None, None

def predicate_from_model(store, model, X, sens, sensors, event, states, predicates,
                         related_sensors):
    valid = True
    sens_values = [states[ts][sens] for ts in event.timestamps]
    mean = np.mean(sens_values)
    noise = 1.27e-3

    var = store[sens]
    for i, ts in enumerate(event.timestamps):
        sens_value = states[ts][sens]
        prediction = model.predict(X[i].reshape(1, -1))[0]
        norm_pred = float((prediction - var.min_val))/(var.max_val - var.min_val)
        norm_sens = float((sens_value - var.min_val))/(var.max_val - var.min_val)
        #error = math.fabs(prediction - sens_value)
        error = math.fabs(norm_pred - norm_sens)
        if error >= noise:
            valid = False
            break

    if valid:
        pred_gt = Predicate(sens, ">", model=model, num_value=mean, error=noise)
        pred_lt = Predicate(sens, "<", model=model, num_value=mean, error=noise)
        if sens not in predicates:
            predicates[sens] = {GT : list(), LS: list()}
        predicates[sens][GT].append(pred_gt)
        predicates[sens][LS].append(pred_lt)
        related_sensors = related_sensors.union(get_correlated_event(sens, sensors, model))

def get_other_sens_values(state, sens, sensors):
    return [state[k] for k in sensors if k != sens]

def get_correlated_event(sens, sensors, model):
    related = set()
    other_sens = [x for x in sensors if x != sens]
    assert len(other_sens) == len(model.coef_)
    for i, coef in enumerate(model.coef_):
        if coef != 0:
            related.add(other_sens[i])
    return related

def is_turn_on_event(from_value, to_value):
    return ((from_value == OFF and to_value == ON) or
            (from_value == OFFZ and to_value == ON))

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
                if((curr_value == OFFZ and values[var] == OFF) or
                   curr_value == OFF and values[var] == OFFZ):
                    continue

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
    generate_actuators_predicates(actuators, store, predicates)

    events = retrieve_update_timestamps(store.discrete_vars(), data)

    generate_sensors_predicates(sensors, store, events, data, predicates)

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
