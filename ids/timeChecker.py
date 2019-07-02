#!/usr/bin/env python3
import sys
import yaml
import collections
import math
import threading
from datetime import datetime, timedelta
import pickle
import pdb

import numpy as np
from scipy import stats

from welford import Welford
from utils import ProcessVariable, randomName
from reqChecker import Checker

ValueTS = collections.namedtuple('ValueTS', ['value', 'ts'])
DIST = 0.01

class TransitionMatrix(object):

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
        self.header = variable.limit_values
        self.historic_val = []
        # map value -> position to row or column of the value in the matrix
        self.val_pos = {}
        self.transitions = self.compute_transition(variable.limit_values)
        self.last_value = None

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
        s = " "
        for val in self.header:
            s += " {}".format(val)
        s += "\n"

        for index, val in enumerate(self.header):
            for v in self.transitions[index]:
                s += " {}".format(v)
            s += "\n"
        return s

    def __str__(self):
        s = "\n".join([str(self.header), str(self.transitions)])
        return s

    def __repr__(self):
        return self.__str__()

    def same_value(self, val1, val2, pv):
        return pv.normalized_dist(val1, val2) <= DIST

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

    def check_transition_time(self, newval, oldval, elapsed_time, pv):

        pdb.set_trace()
        row = self.val_pos[oldval]
        column = self.val_pos[newval]
        expected = self.transitions[row][column]
        res = expected == -1
        if not res:
            z = (elapsed_time - expected.M)/expected.k
            # How likely a elapsed time diff from the mean to be from the same 
            # group of observation
            prob_same = 1 - stats.norm.cdf(z)
            res = 0.05 > prob_same
        return res

    def compute_transition_time(self, newval, ts, pv):
        for crit_val in self.header:
            if self.same_value(newval, crit_val, pv):
                if self.last_value and self.last_value.value != crit_val:
                    elapsed_time = (ts - self.last_value.ts).total_seconds()
                    res = self.check_transition_time(crit_val, self.last_value.value,
                                                     elapsed_time, pv)
                    if res:
                        print("Suspect transitions from {}".format(pv.name))
                    self.last_value = ValueTS(value=crit_val, ts=ts)
                elif not self.last_value:
                    self.last_value = ValueTS(value=crit_val, ts=ts)
                elif self.last_value.value == crit_val:
                    pass

                break
            elif newval < crit_val:
                break
                    
    @Decorators
    def update_transition_matrix(self, pv):
        current_val = None
        current_ts = None
        for val, ts in self.historic_val:
            if current_val is None and current_ts is None:
                current_val = self.find_crit_val(val, pv)
                current_ts = ts

            elif self.same_value(current_val, val, pv):
                current_ts = ts

            elif (not self.same_value(current_val, val, pv) and
                  current_val is not None and
                  current_ts is not None):
                new_current_val = self.find_crit_val(val, pv)
                elapsed_time = (ts - current_ts).total_seconds()
                row = self.val_pos[current_val]
                column = self.val_pos[new_current_val]
                if self.transitions[row][column] == -1:
                    self.transitions[row][column] = Welford(elapsed_time)
                else:
                    self.transitions[row][column](elapsed_time)
                current_val = new_current_val
                current_ts = ts

    def compute_elapsed_time(self):
        elapsed_time = []
        for i in range(len(self.historic_val) - 1):
            cur = self.historic_val[i].ts
            nextframe = self.historic_val[i+1].ts
            elapsed_time.append((nextframe - cur).total_seconds())
        return elapsed_time

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

    def __init__(self, descFile, store, network=False, frameSize=10):
        Checker.__init__(self, descFile, store, network)
        self.done = False
        self.frame_size = timedelta(seconds=frameSize)
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

    def get_values_timestamp(self):

        for state in self.store:
            ts = state['timestamp']
            for name, val in state.items():
                if name != 'timestamp':
                    matrix = self.matrices[name]
                    pv = self.vars[name]
                    matrix.add_value(val, ts, pv)

    def compute_matrices(self):
        for name, var in self.vars.items():
            if name != 'timestamp':
                self.matrices[name].update_transition_matrix(var)

    def detect_suspect_transition(self):
        for state in self.store:
            ts = state['timestamp']
            for name, val in state.items():
                if name != 'timestamp':
                    matrix = self.matrices[name]
                    pv = self.vars[name]
                    matrix.compute_transition_time(val, ts, pv)

    def compute_frame(self, detection=False):
        finish_run = {}
        current_cond = {}

        while True:
            msg = self.store.get()
            self.done = isinstance(msg, str)
            if self.done:
                break

            key = msg.key()

            if key not in finish_run:
                finish_run[key] = False

            pv = self.get_variable(msg, key)

            if pv.first is None:
                pv.last_transition = msg.res_timestamp
                pv.first = msg.res_timestamp

            frame = self.get_transition_matrix(key, pv, detection)

            pv.current_ts = msg.res_timestamp
            pv.value = msg.value

            if key not in current_cond and not finish_run[key]:
                begin = pv.first
                end = pv.first + self.frame_size
                cond = (begin, end)
                current_cond[key] = cond
                frame.add_value(msg.value, msg.res_timestamp)

            elif finish_run[key]:
                frame.add_value(msg.value, msg.res_timestamp)

            elif key in current_cond:
                begin, end = current_cond[key]
                if pv.current_ts > end:
                    new_frame = TransitionMatrix(pv)
                    new_frame.add_value(msg.value, msg.res_timestamp)
                    if not detection:
                        self.map_var_frame[key].append(new_frame)
                    else:
                        self.detection_cond[key].append(new_frame)
                    pv.first = msg.res_timestamp
                    pv.last_transition = msg.res_timestamp
                    finish_run[key] = True
                    del current_cond[key]
                else:
                    frame.add_value(msg.value, msg.res_timestamp)

            if all(finish_run.values()):
                break

    def update_condition(self, nbr_frame):
        for key, frames in self.map_var_frame.items():
            cond = TimeCond()
            for frame in frames[0:nbr_frame]:
                cond.add_expected_value(frame.nbr_transition())
                elapsed_time = frame.compute_elapsed_time()
                cond.add_expected_avg_var(elapsed_time)
            self.map_pv_cond[key] = cond

    def distance_matrix(self, a, b):
        acc = 0
        for i in range(len(a.header)):
            for j in range(len(b.header)):
                val1 = a.transitions[i][j]
                val2 = b.transitions[i][j]
                acc += (val1 - val2)**2
        return math.sqrt(acc)

    def check_condition(self):
        for key, frames in self.detection_cond.items():
            if key not in self.map_pv_cond:
                print("Alert for unknown PV {}".format(key))
                continue

            cond = self.map_pv_cond[key]
            #pdb.set_trace()
            frame = frames.pop(0)
            matrices = self.map_var_frame[key]
            try:
                dist = min([self.distance_matrix(frame, x) for x in matrices])
                print("Key:{}, Distance: {}".format(key, dist))
            except ValueError as err:
                print(err)

            res = cond.test_cond(frame)
            if not res:
                print("Alert for {}\n".format(key))
                print("Got: {}\n".format(repr(frame)))
                print("Expected: {}\n".format(cond))

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
        """
        i = 0
        nbr_frame = 5
        print("Creating condition")
        while not self.done and i < nbr_frame:
            self.compute_frame()
            i += 1
        pdb.set_trace()
        self.update_condition(nbr_frame)

        print("Running detection")
        while not self.done:
            self.compute_frame(True)
            self.check_condition()
            break
        """
