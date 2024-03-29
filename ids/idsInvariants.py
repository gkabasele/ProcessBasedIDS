import re
import ast
import pdb
from collections import OrderedDict
from collections import Counter
from datetime import datetime
from timeit import default_timer as timer
import numpy as np
import operator
import predicate as pred
from utils import TS
from itemset import sensor_predicates, actuator_predicates, get_feature_sensors

IMPLY = "->"
PRED_ID = r"(?P<pred>\d+):\[\(\d+\) (?P<varname>\w+)"

class Invariant(object):

    def __init__(self, identifier, cause, effect):
        self.cause = cause
        self.effect = effect
        self.identifier = identifier

    def is_valid(self, satisfied_predicates, debug=False):
        #FIXME the argument will be sorted by id

        if debug:
            pdb.set_trace()

        for predicate in self.cause:
            if predicate.id in satisfied_predicates:
                continue
            else:
                return True, predicate.id

        for predicate in self.effect:
            if predicate.id in satisfied_predicates:
                continue
            else:
                return False, predicate.id
        return True, self

    def __str__(self):
        return "{}->{}".format(self.cause, self.effect)

    def __repr__(self):
        return self.__str__()

class IDSInvariant(object):

    def __init__(self, mapping_id_pred, invariants, filename):
        # ID->predicates
        self.invariants = self.create_invariants(mapping_id_pred,
                                                 invariants)
        self.malicious_activities = OrderedDict()
        self.filehandler = open(filename, "w+")
        self.elapsed_time_per_computation = list()

    def create_invariants(self, mapping_id_pred, invariantsfile):
        invariants = []
        with open(invariantsfile, "r") as fname:
            for i, line in enumerate(fname):
                tmp_cause, tmp_effect = line.split(IMPLY)
                cause = ast.literal_eval(tmp_cause)
                effect = ast.literal_eval(tmp_effect)

                cause_pred = [mapping_id_pred[pred_id] for pred_id in cause]
                effect_pred = [mapping_id_pred[pred_id] for pred_id in effect]

                invariant = Invariant(i, cause_pred, effect_pred)
                invariants.append(invariant)

        return invariants

    def get_satisfied_predicates(self, state, sensors, predicates, mapping_id_pred):
        tmp_pred = []
        for varname, val in state.items():
            if varname != "timestamp":
                try:
                    if pred.ON in predicates[varname]:
                        actuator_predicates(varname, val, predicates,
                                            tmp_pred, mapping_id_pred)
                    else:
                        sensor_predicates(state, sensors, varname, val,
                                          predicates, tmp_pred,
                                          mapping_id_pred, False)
                except KeyError:
                    # A variable that must be ignored
                    pass

        satisfied_pred = []
        for items in tmp_pred:
            varname, cond, index = items
            p = predicates[varname][cond][index]
            satisfied_pred.append(p.id)
        satisfied_pred.sort()
        return satisfied_pred

    def valid_state(self, state, sensors, predicates, mapping_id_pred):
        satisfied_pred = self.get_satisfied_predicates(state, sensors,
                                                       predicates,
                                                       mapping_id_pred)
        invalid = []
        for i, invariant in enumerate(self.invariants):

            # Line for Debug
            #if i == 129 or i == 254:
            #    features_d = [state[k] for k in sensors if k != "dpit301"]
            #    features_p = [state[k] for k in sensors if k != "pit501"]
            #    dpit_features = np.array(features_d).reshape(1, -1)
            #    pit_features = np.array(features_p).reshape(1, -1)
            #    pdb.set_trace()

            res, failed_pred_id = invariant.is_valid(satisfied_pred)
            if not res:
                if len(invalid) == 0:
                    self.write(state[TS], state)

                msg = "\t id:{} {}\n".format(invariant.identifier,
                                             invariant.effect)
                self.filehandler.write(msg)

                failed_pred = mapping_id_pred[failed_pred_id]

                if type(failed_pred) is pred.PredicateEvent:

                    got_value = state[failed_pred.varname]
                    if failed_pred.operator == operator.eq:
                        exp_value = failed_pred.value
                    else:
                        features = np.array(get_feature_sensors(state, sensors, failed_pred.varname)).reshape(1, -1)
                        if failed_pred.operator == operator.lt:
                            exp_value = failed_pred.model.predict(features)[0] - failed_pred.error

                        elif failed_pred.operator == operator.gt:
                            exp_value = failed_pred.model.predict(features)[0] + failed_pred.error


                    msg = "pred_id:{} got: {}, expected: {}{}\n\n".format(failed_pred_id, got_value, failed_pred.operator, exp_value)
                    self.filehandler.write(msg)
                self.add_malicious_activities(invariant.identifier, failed_pred_id, state[TS])
                invalid.append(i)

        if len(invalid) != 0:
            return invalid

    def add_malicious_activities(self, invariant, failed_pred_id, ts):
        time_key = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
        if time_key in self.malicious_activities:
            self.malicious_activities[time_key].add((invariant, failed_pred_id))
        else:
            self.malicious_activities[time_key] = set()
            self.malicious_activities[time_key].add((invariant, failed_pred_id))

    def get_vars_alerts_hist(self):
        alert_occurence = [t[0] for invariants in self.malicious_activities.values() for t in invariants]
        c = Counter(alert_occurence)
        total = sum(c.values(), 0.0)
        for key in c:
            c[key] /= total
        return c

    def export_detected_atk(self, filename):
        with open(filename, "w") as f:
            for k, v in self.malicious_activities.items():
                f.write("[{}] {}\n".format(k, v))

    def run_detection(self, store, sensors, predicates, mapping_id_pred):
        start_timer = timer()
        for i, state in enumerate(store):
            if i % 3600 == 0:
                end_timer = timer()
                print("Elasped time: {}".format(end_timer - start_timer))
                self.elapsed_time_per_computation.append((end_timer - start_timer))
                print("IDS Invariant Staring state{} of  {}".format(i, len(store)))

            self.valid_state(state, sensors, predicates, mapping_id_pred)

    def write(self, timestamp, line):
        self.filehandler.write("[{}] {}\n".format(timestamp, line))

    def close(self):
        self.filehandler.close()
