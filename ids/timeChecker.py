#!/usr/bin/env python3
import sys
import yaml
import collections
import math
import threading
from datetime import datetime, timedelta
import pickle
import pdb
from copy import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import stats

from utils import ProcessVariable, randomName
from reqChecker import Checker
from timePattern import TimePattern

ValueTS = collections.namedtuple('ValueTS', ['value', 'ts'])
DIST = 0.01
DIFF = 0.05

class TransitionMatrix(object):

    DIFF = "Different"
    SAME = "Same"
    UNKNOWN = "Unknown"
    UNEXPECT = "Unexpected"
    ILLEGAL = "Illegal"

    class Decorators(object):
        def __init__(self, f):
            self.func = f

        def __get__(self, instance, cls=None):
            self._instance = instance
            return self

        def __call__(self, *args, **kwargs):

            if len(self._instance.transitions) != 0:
                res = self.func(self._instance, *args, **kwargs)
            else:
                raise ValueError('Cannot perform computation on transitions Matrix')
            return res


    def __init__(self, variable):
        self.header = self.compute_header(variable)
        self.name = variable.name
        self.historic_val = []
        # map value -> position to row or column of the value in the matrix
        self.val_pos = {}
        self.transitions = self.compute_transition(self.header)
        self.last_value = None
        self.last_val_train = None

    def compute_header(self, variable):
        header = []
        start_index = 0
        end_index = 0
        values = variable.limit_values
        i = 0
        while i < len(values):
            if i == 0:
                header.append(values[i])
            else:
                if self.same_value(values[start_index], values[i], variable):
                    pass
                else:
                    header.append(values[i])
                    start_index = i
            i += 1

        return header

    def compute_transition(self, values):
        transitions = []
        for index, val in enumerate(values):
            b = [-1] * len(values)
            b[index] = 0
            transitions.append(b)
            self.val_pos[val] = index

        """
        a = np.array(transitions)
        return np.reshape(a, (len(values), (len(values))))
        """
        return transitions

    def display_matrix(self):
        s = ""
        for row in range(len(self.header)):
            from_val = self.header[row]
            for column in range(len(self.header)):
                to_val = self.header[column]
                s += "{}->{}: {}\n".format(from_val, to_val, 
                                           self.transitions[row][column])
        return s

    def __str__(self):
        s = "\n".join([str(self.header), str(self.transitions)])
        return s

    def __repr__(self):
        return self.__str__()

    def same_value(self, val1, val2, pv):
        return pv.normalized_dist(val1, val2) <= DIST

    def great_diff(self, val1, val2, pv):
        return pv.normalized_dist(val1, val2) >= DIFF

    def nbr_transition(self):
        return len(self.historic_val) - 1

    def compute_change_prob(self, pv):
        nbr_seq = self.nbr_transition()
        change = 0
        for i in range(len(self.historic_val) - 1):
            cur_val = self.historic_val[i].value
            next_val = self.historic_val[i+1].value
            if not self.same_value(cur_val, next_val, pv):
                change += 1
        return change/nbr_seq

    def add_value(self, val, ts, pv):
        for crit_val in self.header:
            if self.same_value(val, crit_val, pv):
                v = ValueTS(value=val, ts=ts)
                self.historic_val.append(v)
                break
            elif val < crit_val:
                break

    def find_crit_val(self, val, pv):
        for crit_val in self.header:
            if self.same_value(crit_val, val, pv):
                return crit_val

    def _find_closest(self, elapsed_time, prev, curr, i):
        dist_prev = elapsed_time - prev
        dst = curr - elapsed_time
        if dist < dist_prev:
            return i
        else:
            return i-1

    def find_cluster(self, pattern, elapsed_time):
        cluster = None
        bpoints = pattern.breakpoints
        for i, limit in enumerate(bpoints):
            if elapsed_time <= limit:
                if i == 0:
                    cluster = pattern.clusters[i]
                else:
                    prev = pattern.clusters[i-1]
                    index = self._find_closest(elapsed_time, prev, limit, i)
                    cluster = pattern.clusters[index]
                break
            else:
                if i == len(bpoints)-1:
                    cluster = pattern.clusters[i+1]
        return cluster

    def check_transition_time(self, newval, oldval, elapsed_time, pv):
        row = self.val_pos[oldval]
        column = self.val_pos[newval]
        expected = self.transitions[row][column]
        if expected == -1 or expected == 0:
            return TransitionMatrix.UNKNOWN, expected

        cluster = self.find_cluster(expected, elapsed_time)
        print("Elapsed: {}, Cluster:{}".format(elapsed_time, cluster))
        z = (elapsed_time - cluster.mean)/cluster.std
        if abs(z) > 3:
            return TransitionMatrix.DIFF, cluster
        else:
            return TransitionMatrix.SAME, cluster
        """
        #z = (elapsed_time - cluster.mean)/cluster.k
        # How likely a elapsed time diff from the mean to be from the same
        # group of observation
        prob_same = 1 - stats.norm.cdf(z)
        if prob_same < 0.05:
            return TransitionMatrix.DIFF, cluster
        else:
            return TransitionMatrix.SAME, cluster
        """
    def compute_transition_time(self, newval, ts, pv):
        if not self.same_value(newval, self.header[0], pv) and newval < self.header[0]:

            print("[{}][{}] Unexpected value for {}, got {}".format(ts, TransitionMatrix.UNEXPECT,
                                                                    pv.name, newval))
            if self.last_value is None or self.last_value.value != "-inf":
                self.last_value = ValueTS("-inf", ts)
            return

        elif not self.same_value(newval, self.header[-1], pv) and newval > self.header[-1]:

            print("[{}][{}] Unexpected value for {}, got {}".format(ts, TransitionMatrix.UNEXPECT,
                                                                    pv.name, newval))
            if self.last_value is None or self.last_value.value != "inf":
                self.last_value = ValueTS("inf", ts)
            return

        for crit_val in self.header:
            if self.same_value(newval, crit_val, pv):
                if self.last_value is not None:
                    elapsed_time = (ts - self.last_value.ts).total_seconds()
                    if self.last_value.value == "inf" or self.last_value.value == "-inf":
                        print("[{}][{}]transition from illegal position for {} {}->{},  {}".format(ts,
                                                                                              TransitionMatrix.ILLEGAL,
                                                                                              pv.name,
                                                                                              self.last_value.value,
                                                                                              crit_val,
                                                                                              elapsed_time))
                    else:
                        res, expected = self.check_transition_time(crit_val, self.last_value.value,
                                                                   elapsed_time, pv)
                        if res == TransitionMatrix.DIFF or res == TransitionMatrix.UNKNOWN:
                            print("[{}][{}]transitions for {} {}->{}, expected: {}, got:{}".format(ts, res, pv.name,
                                                                                                   self.last_value.value
                                                                                                   , crit_val, expected,
                                                                                                   elapsed_time))
                    if self.last_value.value != crit_val:
                        self.last_value = ValueTS(value=crit_val, ts=ts)
                else:
                    self.last_value = ValueTS(value=crit_val, ts=ts)
                break
            elif newval < crit_val:
                break
    @Decorators
    def update_transition_matrix(self, value, ts, pv):
        for crit_val in self.header:
            if self.same_value(value, crit_val, pv):
                if self.last_val_train is not None:
                    elapsed_time = (ts - self.last_val_train.ts).total_seconds()
                    row = self.val_pos[self.last_val_train.value]
                    column = self.val_pos[crit_val]
                    if self.transitions[row][column] == -1 or self.transitions[row][column] == 0:
                        self.transitions[row][column] = TimePattern()

                    self.transitions[row][column].update(elapsed_time)

                    if self.last_val_train.value != crit_val:
                        self.last_val_train = ValueTS(value=crit_val, ts=ts)
                else:
                    self.last_val_train = ValueTS(value=crit_val, ts=ts)

                break

            elif value < crit_val:
                break

    def compute_elapsed_time(self):
        elapsed_time = []
        for i in range(len(self.historic_val) - 1):
            cur = self.historic_val[i].ts
            nextframe = self.historic_val[i+1].ts
            elapsed_time.append((nextframe - cur).total_seconds())
        return elapsed_time

    def compute_clusters(self):
        try:
            #if self.name == "lit101":
            #    pdb.set_trace()
            for row in range(len(self.header)):
                for column in range(len(self.header)):
                    entry = self.transitions[row][column]
                    if not isinstance(entry, int):
                        entry.create_clusters()
        except np.linalg.LinAlgError:
            pdb.set_trace()


class TimeCond(object):

    def __init__(self):
        self.expected_values = set()
        self.avg_elapsed_val = set()
        self.var_elapsed_val = set()

    def add_expected_value(self, val):
        self.expected_values.add(val)

    def add_expected_avg_var(self, elapsed_time):
        if len(elapsed_time) != 0:
            avg = np.average(elapsed_time)
            var = np.var(elapsed_time)
            self.avg_elapsed_val.add(avg)
            self.var_elapsed_val.add(var)
        else:
            self.avg_elapsed_val.add(-1)
            self.var_elapsed_val.add(-1)

    def compute_t(self, avg, var, ex_avg, n):
        if n > 0:
            t_score = (avg - ex_avg)/math.sqrt(var/n)
            df = n - 1
            alpha = 0.01
            crit_byte = stats.t.ppf(1-alpha, df=df)
            return t_score < crit_byte

    def test_cond(self, frame):
        value = frame.nbr_transition()
        elapsed_time = frame.compute_elapsed_time()
        res = value in self.expected_values
        if res:
            if len(elapsed_time) == 0:
                avg, var = -1, -1
            else:
                avg = np.average(elapsed_time)
                var = np.var(elapsed_time)

            for ex_avg in self.avg_elapsed_val:
                if avg == -1 or ex_avg == -1:
                    res = avg == ex_avg
                else:
                    res = self.compute_t(avg, var, ex_avg, len(elapsed_time))
                if res:
                    return res
        return res

    def __str__(self):
        return "Val: {}, Avg: {}, Var: {}".format(self.expected_values,
                                                  self.avg_elapsed_val,
                                                  self.var_elapsed_val)
    def __repr__(self):
        return self.__str__()

class TimeChecker(Checker):

    def __init__(self, descFile, store, detection_store=None, network=False,
                 frameSize=10):
        Checker.__init__(self, descFile, store, network)
        self.done = False
        self.frame_size = timedelta(seconds=frameSize)
        self.detection_store = detection_store
        self.messages = {}

        self.map_pv_cond = {}
        self.map_var_frame = {}

        self.detection_cond = {}

        self.matrices = self.create_matrices()

    def create_matrices(self):
        matrices = {}
        for name, variable in self.vars.items():
            matrices[name] = TransitionMatrix(variable)
        return matrices

    def create_var(self, host, port, kind, addr):
        pv = ProcessVariable(host, port, kind, addr, name=randomName())
        self.vars[pv.name] = pv
        self.map_key_name[pv.key()] = pv.name
        return pv

    def get_variable(self, msg, key):
        if key not in self.map_key_name:
            pv = self.create_var(msg.host, msg.port, msg.kind, msg.addr)
        else:
            name = self.map_key_name[key]
            pv = self.vars[name]
        return pv

    def get_transition_matrix(self, key, variable, detection):
        if not detection:
            if key not in self.map_var_frame:
                self.map_var_frame[key] = [TransitionMatrix(variable)]
            return self.map_var_frame[key][-1]
        else:
            if key not in self.detection_cond:
                self.detection_cond[key] = [TransitionMatrix(variable)]
            return self.detection_cond[key][-1]

    def fill_matrices(self):

        for state in self.store:
            ts = state['timestamp']
            for name, val in state.items():
                if name != 'timestamp' and name != 'normal/attack':
                    matrix = self.matrices[name]
                    pv = self.vars[name]
                    matrix.update_transition_matrix(val, ts, pv)

        for name in self.vars:
            self.matrices[name].compute_clusters()

    def detect_suspect_transition(self):
        if self.detection_store is not None:
            for state in self.detection_store:
                ts = state['timestamp']
                for name, val in state.items():
                    if name != 'timestamp' and name != 'normal/attack':
                        matrix = self.matrices[name]
                        pv = self.vars[name]
                        matrix.compute_transition_time(val, ts, pv)

    def display_message(self):
        s = ""
        for k, v in self.messages.items():
            s += "{}->{}\n".format(k, v)
        return s

    def display_dict(self, d):
        s = ""
        for k, v in d.items():
            s += "{}->{}\n\n".format(k, v)
        return s

    def run(self):
        self.fill_matrices()
        print("Passing in detection mode")
        pdb.set_trace()
        self.detect_suspect_transition()
