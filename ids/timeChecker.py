#!/usr/bin/env python3
import sys
import yaml
import collections
import math
import threading
from datetime import datetime, timedelta
import pickle
import pdb
from collections import OrderedDict
from copy import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import stats

import utils
from reqChecker import Checker
from timePattern import TimePattern

ValueTS = collections.namedtuple('ValueTS', ['value', 'start', 'end'])


# Threshold to compare the times
BIG_THRESH = 1
EQ_THRESH = 0.1

# Threshold to compare times, noisy precision
BIG_NOISY_THRESH = 0.25
EQ_NOISY_THRESH = 0.05

# Number of outliers accpeted to consolidate the patterns
N_OUTLIER = 3
N_CLUSTERS = 2

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


    def __init__(self, variable, noisy=True):
        self.noisy = noisy
        self.header = self.compute_header(variable)
        self.name = variable.name
        self.historic_val = []

        # map value -> position to row or column of the value in the matrix
        self.val_pos = {}
        self.transitions = self.compute_transition(self.header)
        #Last value of the variable in the detection phase
        self.last_value = None
        # Last value of the variable in the training phase
        self.last_val_train = None
        # 
        self.start_same_train = None
        #
        self.end_same_train = None
        # Variable to store whether or not the training phase is done since
        # we have reached last timestamp of dataset. This is needed to trigger
        # a last update of the matrix
        self.end_time = None
        # flag to know if an elapsed time must be computed
        self.computation_trigger = False
        # flag to know when to detect
        self.detection_trigger = False

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

    # Create matrix of transitions
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
        if self.noisy:
            return pv.normalized_dist(val1, val2) <= utils.DIST
        else:
            return val1 == val2

    def great_diff(self, val1, val2, pv):
        if self.noisy:
            return pv.normalized_dist(val1, val2) >= utils.DIFF
        else:
            return val1 != val2

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

    def add_malicious_activities(self, malicious_activities, ts, pv):
        time_key = datetime(ts.year, ts.month, ts.day,
                            ts.hour, ts.minute, ts.second)
        if time_key not in malicious_activities:
            malicious_activities[time_key] = set()

        malicious_activities[time_key].add(pv)


    def find_crit_val(self, val, pv):
        for crit_val in self.header:
            if self.same_value(crit_val, val, pv):
                return crit_val

    def _find_closest(self, elapsed_time, prev, curr, i):
        dist_prev = elapsed_time - prev.mean
        dist = curr - elapsed_time
        if dist < dist_prev:
            return i
        else:
            return i-1

    def find_cluster(self, pattern, elapsed_time):
        if len(pattern.clusters) == 1:
            return pattern.clusters[0]
        else:
            return pattern.get_cluster(elapsed_time)

    # Comparison for testing with the simulator
    def is_close_time(self, t1, t2):
        diff = abs(t1 - t2)
        return diff <= EQ_THRESH

    def is_big_time(self, t1, t2):
        diff = abs(t1 -t2)
        return diff >= BIG_THRESH

    def is_close_time_noisy(self, t1, cluster):
        if cluster.max_val == cluster.min_val:
            return t1 == cluster.mean

        diff_max = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                         t1, cluster.max_val)
        diff_min = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                         t1, cluster.min_val)

        diff = min(diff_max, diff_min)
        return diff >= EQ_NOISY_THRESH

    def is_big_time_noisy(self, t1, cluster):
        if cluster.max_val == cluster.min_val:
            return t1 == cluster.mean
        diff_max = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                         t1, cluster.max_val)
        diff_min = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                         t1, cluster.min_val)

        diff = min(diff_max, diff_min)
        return diff >= BIG_NOISY_THRESH

    def outlier_handler(self, column, row, elapsed_time, cluster):
        pattern = self.transitions[row][column]
        nbr_outlier = 0
        for c in pattern.clusters:
            if c.std == 0 and c.k < 3:
                nbr_outlier += 1

        #Check if cluster is too far from outlier to consolidate cluster or not
        if (nbr_outlier >= N_OUTLIER and
                not self.is_big_time(elapsed_time, cluster.mean)):
            cluster(elapsed_time)
            return TransitionMatrix.SAME, cluster
        else:
            return TransitionMatrix.DIFF, cluster

    def outlier_handler_noisy(self, column, row, elapsed_time, cluster):
        pattern = self.transitions[row][column]
        nbr_outlier = 0
        for c in pattern.clusters:
            if c.std == 0 and c.k < 3:
                nbr_outlier += 1

        if (nbr_outlier >= N_OUTLIER and
                not self.is_big_time_noisy(elapsed_time, cluster)):
            cluster(elapsed_time)
            return TransitionMatrix.SAME, cluster
        else:
            return TransitionMatrix.DIFF, cluster

    def check_transition_time(self, newval, oldval, elapsed_time, pv):
        row = self.val_pos[oldval]
        column = self.val_pos[newval]
        expected = self.transitions[row][column]
        #if pv.name == "tankFinal" or pv.name == "silo2" or pv.name == "wagonlidOpen":
        #    pdb.set_trace()
        if expected == -1 or expected == 0:
            return TransitionMatrix.UNKNOWN, expected

        if len(expected.clusters) < N_CLUSTERS:
            # Too few information about that transition to give an opinion
            # By default we accept the transition
            return TransitionMatrix.SAME, None

        cluster = self.find_cluster(expected, elapsed_time)

        ## Test with the simulator##
        if not self.noisy:
            if cluster.std == 0:
                if cluster.k >= 3:
                    if self.is_close_time(elapsed_time, cluster.mean):
                        return TransitionMatrix.SAME, cluster
                    else:
                        return TransitionMatrix.DIFF, cluster
                else:
                    if self.is_close_time(elapsed_time, cluster.mean):
                        return TransitionMatrix.SAME, cluster
                    else:
                        return self.outlier_handler(column, row, elapsed_time, cluster)
            else:
                # Chebyshev's inequality
                z = (elapsed_time - cluster.mean)/cluster.std
                ztest = abs(z) <= 3
                close = self.is_close_time(elapsed_time, cluster.mean)
                if ztest or close:
                    return TransitionMatrix.SAME, cluster
                else:
                    return TransitionMatrix.DIFF, cluster
        else:

            if cluster.std == 0:
                #FIXME number of time a value has been seen consider it meaningfull
                # If the exact same elapsed_time keep repeating itself then it is 
                # constant so the value should always be the same
                # Otherwise there was an outlier in during the training process

                # Min/max value should be the local value (cluster) not global
                if  cluster.k >= 3:
                    if self.is_close_time_noisy(elapsed_time, cluster):
                        return TransitionMatrix.SAME, cluster
                    else:
                        return TransitionMatrix.DIFF, cluster
                else:
                    if self.is_close_time_noisy(elapsed_time, cluster):
                        return TransitionMatrix.SAME, cluster
                    else:
                        return self.outlier_handler_noisy(column, row, elapsed_time, cluster)

            else:
                z = (elapsed_time - cluster.mean)/cluster.std
                ztest = abs(z) <= 3
                close = self.is_close_time_noisy(elapsed_time, cluster)

                if ztest or close:
                    return TransitionMatrix.SAME, cluster
                else:
                    return TransitionMatrix.DIFF, cluster
                #z = (elapsed_time - cluster.mean)/cluster.k
                ##How likely a elapsed time diff from the mean to be from the same
                ##group of observation
                #prob_same = 1 - stats.norm.cdf(z)
                #if prob_same < 0.05:
                #    return TransitionMatrix.DIFF, cluster
                #else:
                #    return TransitionMatrix.SAME, cluster

    def write_msg(self, filehandler, res, ts, pv_name, got, expected, 
                  malicious_activities,crit_val=None, last_val=None):

        if res == TransitionMatrix.UNEXPECT:
            filehandler.write("[{}][{}] Unexpected value for {}, expected:{}, got {}\n".format(ts, res, pv_name,
                                                                                              expected, got))
            self.add_malicious_activities(malicious_activities, ts, pv_name)

        else:
            filehandler.write("[{}][{}]transitions for {} {}->{}, expected:{}, got:{}\n".format(ts, res, pv_name,
                                                                                                last_val, crit_val,
                                                                                                expected, got))
            self.add_malicious_activities(malicious_activities, ts, pv_name)

    def compute_transition_time(self, newval, ts, pv, filehandler, malicious_activities):
        # Value greater (resp. lower) than max (resp. min)
        if not self.same_value(newval, self.header[0], pv) and newval < self.header[0]:

            self.write_msg(filehandler, TransitionMatrix.UNEXPECT, ts, pv.name,
                           newval, self.header[0], malicious_activities)

        elif not self.same_value(newval, self.header[-1], pv) and newval > self.header[-1]:

            self.write_msg(filehandler, TransitionMatrix.UNEXPECT, ts, pv.name,
                           newval, self.header[-1], malicious_activities)
        
        # If no critical value was found, we need to compute how long the last
        # critical value remained
        found = False
        for crit_val in self.header:
            if self.same_value(newval, crit_val, pv):
                self.detection_trigger = True
                found = True
                if self.last_value is not None:
                    # The value did not change
                    if self.last_value.value == crit_val:
                        self.last_value = ValueTS(value=crit_val,
                                                  start=self.last_value.start,
                                                  end=ts)
                    # The value has changed since last time
                    else:
                        if not pv.is_bool_var():
                            elapsed_trans_t = (ts - self.last_value.end).total_seconds()
                            res, expected = self.check_transition_time(crit_val, self.last_value.value,
                                                                       elapsed_trans_t, pv)

                            if res == TransitionMatrix.DIFF or res == TransitionMatrix.UNKNOWN:
                                self.write_msg(filehandler, res, ts, pv.name, elapsed_trans_t, expected,
                                               malicious_activities, crit_val=crit_val,
                                               last_val=self.last_value.value)
                        # When a value change from one critical value to another without transition
                        # value so we must compute how long a value remained
                        #if pv.is_bool_var():
                        same_value_t = (self.last_value.end - self.last_value.start).total_seconds()
                        res, expected = self.check_transition_time(self.last_value.value,
                                                                   self.last_value.value,
                                                                   same_value_t, pv)

                        if res == TransitionMatrix.DIFF or res == TransitionMatrix.UNKNOWN:
                            self.write_msg(filehandler, res, ts, pv.name, same_value_t, expected,
                                           malicious_activities, crit_val=self.last_value.value,
                                           last_val=self.last_value.value)

                        self.last_value = ValueTS(value=crit_val, start=ts, end=ts)
                else:
                    self.last_value = ValueTS(value=crit_val, start=ts, end=ts)
                break

            if newval < crit_val:
                break


        # There is no more modification than will occur
        if not found or ts == self.end_time:

            if self.detection_trigger and self.last_value is not None:
                self.detection_trigger = False
                same_value_t = (self.last_value.end - self.last_value.start).total_seconds()
                res, expected = self.check_transition_time(self.last_value.value,
                                                           self.last_value.value, same_value_t, pv)
                if res == TransitionMatrix.DIFF or res == TransitionMatrix.UNKNOWN:
                    self.write_msg(filehandler, res, ts, pv.name, same_value_t, expected,
                                   malicious_activities, crit_val=self.last_value.value,
                                   last_val=self.last_value.value)

    # Add the elasped times of each transition.
    @Decorators
    def update_transition_matrix(self, value, ts, pv):
        for crit_val in self.header:
            if self.same_value(value, crit_val, pv):
                if self.last_val_train is not None:
                    row = self.val_pos[self.last_val_train.value]
                    column = self.val_pos[crit_val]

                    if self.transitions[row][column] == -1:
                        self.transitions[row][column] = TimePattern()

                    if self.transitions[row][row] == 0:
                        self.transitions[row][row] = TimePattern()

                    if self.last_val_train.value == crit_val:
                        self.last_val_train = ValueTS(value=crit_val,
                                                      start=self.last_val_train.start,
                                                      end=ts)
                        self.computation_trigger = False
                    else:
                        same_value_t = (self.last_val_train.end - self.last_val_train.start).total_seconds()
                        self.transitions[row][row].update(same_value_t)
                        self.computation_trigger = True
                        elapsed_trans_t = (ts - self.last_val_train.end).total_seconds()
                        self.last_val_train = ValueTS(value=crit_val, start=ts, end=ts)
                        self.transitions[row][column].update(elapsed_trans_t)
                else:
                    self.last_val_train = ValueTS(value=crit_val, start=ts, end=ts)
                    self.computation_trigger = False
                break
            else:
                self.computation_trigger = False
                if value < crit_val:
                    break

        # The case where the end training period was reached and no computation
        # of transition time was done because no transition time was reach
        if ts == self.end_time and not self.computation_trigger:
            row = self.val_pos[self.last_val_train.value]
            same_value_t = (self.last_val_train.end - self.last_val_train.start).total_seconds()
            self.transitions[row][row].update(same_value_t)

    def compute_elapsed_time(self):
        elapsed_time = []
        for i in range(len(self.historic_val) - 1):
            cur = self.historic_val[i].ts
            nextframe = self.historic_val[i+1].ts
            elapsed_time.append((nextframe - cur).total_seconds())
        return elapsed_time

    def compute_clusters(self):
        try:
            for row in range(len(self.header)):
                for column in range(len(self.header)):
                    entry = self.transitions[row][column]
                    if not isinstance(entry, int):
                        entry.create_clusters()
        except ValueError:
            pdb.set_trace()

class TimeChecker(Checker):

    def __init__(self, descFile, filename, store, noisy=True, detection_store=None,
                 network=False, frameSize=10):
                 
        Checker.__init__(self, descFile, store, network)
        self.done = False
        self.frame_size = timedelta(seconds=frameSize)
        self.detection_store = detection_store
        self.messages = {}

        self.noisy = noisy

        self.map_pv_cond = {}
        self.map_var_frame = {}

        self.detection_cond = {}
        self.filehandler = open(filename, "w+")

        self.matrices = self.create_matrices()

        # state considered as activities {<ts>:[target1, target2]}
        self.malicious_activities = OrderedDict()

    def close(self):
        self.filehandler.close()

    def create_matrices(self):
        matrices = {}
        for name, variable in self.vars.items():
            if variable.is_periodic:
                matrices[name] = TransitionMatrix(variable, self.noisy)
        return matrices

    def is_var_of_interest(self, name):
        return name != 'timestamp' and name != 'normal/attack' and self.vars[name].is_periodic

    def basic_detection(self, name, value, ts):
        pv = self.vars[name]
        if (value > pv.max_val and
                not utils.same_value(pv.max_val, pv.min_val, value, pv.max_val, noisy=self.noisy)):
            self.filehandler.write("[{}] Value too high for {} expected:{}, got:{}\n".format(ts, name,
                                                                                             pv.max_val, value))
        elif (value < pv.min_val and
              not utils.same_value(pv.max_val, pv.min_val, value, pv.min_val, noisy=self.noisy)):
            self.filehandler.write("[{}] Value too low for {} expected:{}, got:{}\n".format(ts, name,
                                                                                            pv.min_val, value))
    def fill_matrices(self):
        for i, state in enumerate(self.store):
            ts = state['timestamp']
            for name, val in state.items():
                if self.is_var_of_interest(name):
                    matrix = self.matrices[name]
                    if i == len(self.store) - 1:
                        matrix.end_time = ts
                    pv = self.vars[name]
                    matrix.update_transition_matrix(val, ts, pv)

        for name, val in self.vars.items():
            if val.is_periodic:
                self.matrices[name].compute_clusters()

    def detect_suspect_transition(self):
        if self.detection_store is not None:
            for i, state in enumerate(self.detection_store):
                ts = state['timestamp']
                for name, val in state.items():
                    if self.is_var_of_interest(name):
                        matrix = self.matrices[name]
                        if i == len(self.detection_store) - 1:
                            matrix.end_time = ts
                        pv = self.vars[name]
                        matrix.compute_transition_time(val, ts, pv, self.filehandler,
                                                       self.malicious_activities)
                    elif name != 'timestamp' and name != 'normal/attack':
                        self.basic_detection(name, val, ts)
        else:
            raise(ValueError("No data to run on for detection"))

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
