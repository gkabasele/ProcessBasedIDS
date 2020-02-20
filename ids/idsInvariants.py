import re
import ast
import pdb
import predicate as pred
from utils import TS
from itemset import sensor_predicates, actuator_predicates

IMPLY = "->"
PRED_ID = r"(?P<pred>\d+):\[\(\d+\) (?P<varname>\w+)"

class Invariant(object):

    def __init__(self, cause, effect):
        self.cause = cause
        self.effect = effect

    def is_valid(self, satisfied_predicates, debug=False):
        #FIXME the argument will be sorted by id

        if debug:
            pdb.set_trace()

        for predicate in self.cause:
            if predicate.id in satisfied_predicates:
                continue
            else:
                return True, self

        for predicate in self.effect:
            if predicate.id in satisfied_predicates:
                continue
            else:
                return False, self
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
        self.malicious_activities = set()
        self.filehandler = open(filename, "w+")

    def create_invariants(self, mapping_id_pred, invariantsfile):
        invariants = []
        with open(invariantsfile, "r") as fname:
            for line in fname:
                tmp_cause, tmp_effect = line.split(IMPLY)
                cause = ast.literal_eval(tmp_cause)
                effect = ast.literal_eval(tmp_effect)

                cause_pred = [mapping_id_pred[pred_id] for pred_id in cause]
                effect_pred = [mapping_id_pred[pred_id] for pred_id in effect]

                invariant = Invariant(cause_pred, effect_pred)
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

            res, _ = invariant.is_valid(satisfied_pred)
            if not res:
                if len(invalid) == 0:
                    self.write(state[TS], state)

                self.malicious_activities.add(state[TS])
                msg = "\t id:{}\n".format(invariant.effect)
                self.filehandler.write(msg)
                invalid.append(i)
        if len(invalid) != 0:
            return invalid

    def write(self, timestamp, line):
        self.filehandler.write("[{}] {}\n".format(timestamp, line))

    def close(self):
        self.filehandler.close()
