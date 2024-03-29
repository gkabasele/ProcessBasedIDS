import argparse
import operator
import pdb
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from matplotlib import rc
import scipy.stats as stats
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

DELTA = 'delta'
DIST = 'dist'

class Predicate(object):
    pred_id = 1

    def __init__(self, varname):
        self.id = Predicate.pred_id
        Predicate.pred_id += 1
        self.varname = varname
        self.support = 0

class Model(object):

    def __init__(self, varname, model):
        self.varname = varname
        self.all_proba = []
        self.gmm = model

    def get_valid_predicate(self, value):
        membership_proba = self.gmm.predict_proba(value)
        self.all_proba.append(membership_proba)
        return np.where(membership_proba[0] == max(membership_proba[0]))[0][0]

class PredicateDist(Predicate):
    # This class is only a placeholder for debugging
    def __init__(self, varname, weight, mean, std):
        Predicate.__init__(self, varname)
        self.weight = weight
        self.mean = mean
        self.std = std

    def __str__(self):
        return "(({}) {} w:{},m:{},std:{})".format(self.id, self.varname,
                                                   self.weight, self.mean,
                                                   self.std)

    def __repr__(self):
        return self.__str__()


class PredicateEvent(Predicate):
    ops = {
        "<" : operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">" : operator.gt
        }
    CLOSE_THRESH_INTERCEPT = 0.01
    CLOSE_THRESH_COEF = 0.01

    def __init__(self, varname, operator, bool_value=None, num_value=None,
                 model=None, error=0):

        Predicate.__init__(self, varname)
        try:
            self.operator = PredicateEvent.ops[operator]
            self.op_label = operator
        except KeyError:
            raise KeyError("Unknown operator for a Predicate")


        if bool_value is not None:
            self.value = bool_value
            self.model = model
        elif model is not None and num_value is not None:
            self.model = model
            self.value = num_value
            self.error = error
        else:
            raise ValueError("No model or value for the predicate")


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

    '''
        Function to check if the linear regression of two model are close to each other.
        To do so it compare the intercept and the coefficient of both predicate
    '''
    def close_to_predicate(self, other_model, var):
        if self.model is None:
            raise ValueError("The predicate does not have a model to make the comparaison")

        other_intercept_norm = float((other_model.intercept_ - var.min_val))/(var.max_val - var.min_val)
        self_intercept_norm = float((self.model.intercept_ - var.min_val))/(var.max_val - var.min_val)


        if abs(other_intercept_norm - self_intercept_norm) > PredicateEvent.CLOSE_THRESH_INTERCEPT:
            return False

        if len(self.model.coef_) != len(other_model.coef_):
            return False

        diff = 0
        # Maybe square the difference instead of absolute value
        for x, y in zip(self.model.coef_, other_model.coef_):
            diff += abs(x - y)

        if float(diff)/len(self.model.coef_) > PredicateEvent.CLOSE_THRESH_COEF:
            return False

        return True

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

    def __hash__(self):
        return self.__str__().__hash__()

def get_sensors_updates(sensors, states, store):

    last_values = {k : None for k in sensors if not store[k].ignore}
    updates = {k: [] for k in sensors if not store[k].ignore}

    for i, state in enumerate(states):
        if i != 0:
            for sensor, deltas in updates.items():
                deltas.append(state[sensor] - last_values[sensor])
        for sensor in sensors:
            last_values[sensor] = state[sensor]

    return updates

def plot_GMM(sensor, model, samples):

    color = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'olive', 'cyan', 'gray']
    weights = model.weights_
    means = model.means_
    covars = model.covariances_

    plt.hist(samples, bins=200, histtype='bar', density=True, ec="red", alpha=0.5)
    x_axis = samples.copy().ravel()
    x_axis.sort()
    for i in range(len(means)):
        plt.plot(x_axis, weights[i]*stats.norm.pdf(x_axis, means[i],
                                                   np.sqrt(covars[i])).ravel(), c=color[i])

    plt.rcParams["agg.path.chunksize"] = 10000
    plt.grid()
    plt.title("{}, GMM:{}".format(sensor, len(means)))
    plt.savefig("eval_swat_process/{}_gmm.png".format(sensor))


def generate_sensors_predicates_dist(sensors, store, states, predicates, n_comp=3):
    updates = get_sensors_updates(sensors, states, store)
    for sensor, deltas in updates.items():
        samples = np.array(deltas).reshape(-1, 1)
        model = model_for_sensor(samples, n_comp)
        if sensor not in predicates:
            predicates[sensor] = {DELTA: model}
        else:
            predicates[sensor][DELTA] = Model(sensor, model)

        weights = model.weights_
        means = model.means_
        covars = model.covariances_

        predicates[sensor][DIST] = [PredicateDist(sensor, weights[i], means[i], np.sqrt(covars[i])) for i in range(model.n_components)]

def model_for_sensor(deltas, n_comp):
    models = [None for i in range(n_comp)]
    for i in range(1, n_comp+1):
        models[i-1] = GaussianMixture(n_components=i, init_params='kmeans').fit(deltas)

    bics = [m.bic(deltas) for m in models]
    return models[bics.index(min(bics))]

def generate_actuators_predicates(actuators, store, predicates):

    for var in actuators:
        if not store[var].ignore:
            predicates[var] = {ON : [PredicateEvent(var, "==", ON)],
                               OFF: [PredicateEvent(var, "==", OFF)]}

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
        error = math.fabs(norm_pred - norm_sens)
        if error >= noise:
            valid = False
            break

    if valid:
        if sens not in predicates:
            predicates[sens] = {GT : list(), LS: list()}

        if not has_similar_predicate(sens, model, predicates, var):

            pred_gt = PredicateEvent(sens, ">", model=model, num_value=mean, error=noise)
            pred_lt = PredicateEvent(sens, "<", model=model, num_value=mean, error=noise)

            predicates[sens][GT].append(pred_gt)
            predicates[sens][LS].append(pred_lt)

        related_sensors = related_sensors.union(get_correlated_event(sens, sensors, model))

def has_similar_predicate(sens, model, predicates, var):
    for p in predicates[sens][GT]:
        if p.close_to_predicate(model, var):
            return True
    return False

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

    actuators = store.discrete_monitor_vars()
    sensors = store.continuous_monitor_vars()

    predicates = {}

    generate_actuators_predicates(actuators, store, predicates)

    events = retrieve_update_timestamps(store.discrete_vars(), data)
    generate_sensors_predicates(sensors, store, events, data, predicates)

    generate_sensors_predicates_dist(sensors, store, data, predicates)
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
