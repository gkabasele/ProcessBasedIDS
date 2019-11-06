import re
import ast
import pdb
from utils import TS

IMPLY = "->"
PRED_ID = r"(?P<pred>\d+):\[\(\d+\) (?P<varname>\w+)"

class Invariant(object):

    def __init__(self, cause, effect):
        self.cause = cause
        self.effect = effect

    def is_valid(self, state):
        for pred in self.cause:
            varname = pred.varname
            value = state[varname]
            if pred.is_true(value):
                continue
            else:
                return True, self

        for pred in self.effect:
            varname = pred.varname
            value = state[varname]
            if pred.is_true(value):
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

    def valid_state(self, state):
        invalid = []
        for i, invariant in enumerate(self.invariants):
            res, _ = invariant.is_valid(state)
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
