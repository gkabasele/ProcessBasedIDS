#!/usr/bin/env python3
import collections
import pickle
from collections import OrderedDict, Counter
from datetime import datetime, timedelta
from timeit import default_timer as timer
import pdb
import numpy as np

import utils
import welford
from reqChecker import Checker
from timePattern import TimePattern

ValueTS = collections.namedtuple('ValueTS', ['value', 'start', 'end'])

LOF = "lof"
ISO = "isolation"
SVM = "svm"

# Threshold to compare the times
BIG_THRESH = 1
EQ_THRESH = 0.5
#EQ_THRESH = 0.5
#0.05, 0.1, 0.15, 0.2, 0.5, 0.7

#0.1, 0.05, 0.07, 0.15

# Threshold to compare times, noisy precision
BIG_NOISY_THRESH = 0.25
EQ_NOISY_THRESH = 0.05
#EQ_NOISY_THRESH = 5
# 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1

# Number of outliers accpeted to consolidate the patterns
N_OUTLIER = 3
N_CLUSTERS = 2

SMALL_CLUSTER  = 0.05
UNSTABLE_CLUSTER = 0.3

class TransitionLogEntry(object):
    __slots__ = ['varname', 'from_val', 'to_val', 'start', 'end', 'trans_time', 'step_mean']
   
    def __init__(self, varname, from_val, to_val, start, end,
                 trans_time, step_mean):

        self.varname = varname
        self.from_val = from_val
        self.to_val = to_val
        self.start = start
        self.end = end
        self.trans_time = trans_time
        self.step_mean = step_mean 

    def __str__(self):
        return "{} ({}->{}): {},{} ({},{})".format(self.varname,
                                           self.from_val, self.to_val,
                                           self.start, self.end,
                                           self.trans_time, self.step_mean)
    def __repr__(self):
        return self.__str__()


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


    def __init__(self, variable, strategy, noisy=True):
        self.noisy = noisy
        self.strategy = strategy
        self.min_val = variable.min_val
        self.max_val = variable.max_val
        self.header = [x for x in range(len(variable.limit_values))]
        self.name = variable.name
        self.historic_val = []

        # map value -> position to row or column of the value in the matrix
        self.val_pos = {}
        self.transitions = self.compute_transition()
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

        # flag to know when a process variable has moved from a critical value
        # this is to handle the case where a pv moved from a critical value and
        # then come back to it without passing by another critical value
        self.has_changed = False

        # variable to keep track of the update step
        self.last_exact_val = None
        #self.update_step = list()
        #self.update_step_still = list()
        self.update_step_still = welford.Welford() 
        self.update_step = welford.Welford()

        self.number_test_performed = 0
        # Determine if a transition/stillness is taking to much time
        self.slowing_phase = False
        self.last_check = None

        #variable to control number of detection
        self.control_detection = 0


    # Create matrix of transitions
    def compute_transition(self):
        transitions = []
        for index, val in enumerate(self.header):
            b = [-1] * len(self.header)
            b[index] = 0
            transitions.append(b)
            self.val_pos[val] = index

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
        if not pv.is_bool_var():
            range_val1, _ = pv.digitizer.get_range(val1)
            range_val2, _ = pv.digitizer.get_range(val2)
            return range_val1 == range_val2
        else:
            return val1 == val2

    def is_out_of_bound(self, newval, extrema, pv, mincheck=True):
        if mincheck:
            return newval < extrema - (10*pv.digitizer.width)
        else:
            return newval > extrema + (10*pv.digitizer.width)


    def nbr_transition(self):
        return len(self.historic_val) - 1

    def add_value(self, val, ts, pv):
        for crit_val in self.header:
            if self.same_value(val, crit_val, pv):
                v = ValueTS(value=val, ts=ts)
                self.historic_val.append(v)
                break
            elif val < crit_val:
                break

    def add_malicious_activities(self, malicious_activities, ts, pv, oldval, newval):
        time_key = datetime(ts.year, ts.month, ts.day,
                            ts.hour, ts.minute, ts.second)
        if time_key not in malicious_activities:
            malicious_activities[time_key] = set()

        malicious_activities[time_key].add((pv, oldval, newval))


    def find_crit_val(self, val, pv):
        if not pv.is_bool_var():
            range_index, _ = pv.digitizer.get_range(val)

            for crit_val in self.header:
                if range_index in pv.limit_values[crit_val]:
                    return crit_val
        else:
            for crit_val in self.header:
                if crit_val == val:
                    return crit_val

    def find_in_between_crit_val(self, oldval, newval, pv):
        if not pv.is_bool_var():
            first_index, _ = pv.digitizer.get_range(oldval)
            second_index, _ = pv.digitizer.get_range(newval)

            for crit_val in self.header:
                for v in pv.limit_values[crit_val]:
                    if ((v < first_index and v > second_index) or
                        (v < second_index and v > first_index)):
                        return crit_val
        
    def get_pattern(self, oldval, newval):
        row = self.val_pos[oldval]
        col = self.val_pos[newval]
        return self.transitions[row][col]

    def test_validity_transition(self, elapsed_time, updates, oldval, newval, stillness, debug=False):
        row = self.val_pos[oldval]
        col = self.val_pos[newval]
        pattern = self.transitions[row][col]
        # Unknown transition since there is no pattern associated
        if isinstance(pattern, int) and (pattern == 0 or pattern == -1):
            return TransitionMatrix.UNKNOWN, pattern, None

        if self.strategy == LOF:
            is_outlier, score = pattern.is_outlier_hdbscan(elapsed_time, updates, stillness, debug)
        elif self.strategy == ISO:
            is_outlier, score = pattern.is_outlier_forest(elapsed_time, updates, stillness, debug)
        elif self.strategy == SVM:
            is_outlier, score = pattern.is_outlier_svm(elapsed_time, updates, stillness, debug)
        else:
            raise ValueError("Unknown strategy: {}".format(self.strategy))

        # score is none if no cluster were defined for that transition
        if is_outlier:
            return TransitionMatrix.DIFF, score, pattern.threshold
        else:
            return TransitionMatrix.SAME, score, pattern.threshold

    def write_msg(self, filehandler, res, ts, pv_name, got, expected,
                  malicious_activities):

        filehandler.write("[{}][{}] Unexpected value for {}, expected:{}, got {}\n".format(ts, res, pv_name,
                                                                                           expected, got))
        self.add_malicious_activities(malicious_activities, ts, pv_name, None, None)


    def write_msg_diff(self, filehandler, res, ts, pv_name, elapsed_time, update_mean,
                       malicious_activities, score, thresh, crit_val, last_val):


        filehandler.write("[{}][{}]transitions for {} {}->{}, Values:[{} {}], score:{}, thresh:{}\n".format(ts, res, pv_name,
                                                                                                            last_val, crit_val,
                                                                                                            elapsed_time, update_mean,
                                                                                                            score, thresh))
        self.add_malicious_activities(malicious_activities, ts, pv_name, last_val, crit_val)


    def write_msg_unknown(self, filehandler, res, ts, pv_name, malicious_activities, crit_val, last_val):

        filehandler.write("[{}][{}]transitions for {} {}->{}\n".format(ts, res, pv_name, last_val, crit_val))

        self.add_malicious_activities(malicious_activities, ts, pv_name, last_val, crit_val)


    def perform_detection_test(self, filehandler, ts, pv, malicious_activities,
                               elapsed_val, update_mean, oldval, newval, stillness=False):

        res, score, threshold = self.test_validity_transition(elapsed_val,
                                                              update_mean,
                                                              oldval,
                                                              newval,
                                                              stillness)
        # TEST actuators intensive alert
        #if (res == TransitionMatrix.DIFF or
        #        res == TransitionMatrix.UNKNOWN):

        #    rerun = (pv.name in ["mv303", "mv301", "p203"] and oldval == 1 and newval == 1)
        #    if rerun:
        #        print(pv.name)
        #        self.test_validity_transition(elapsed_val, update_mean,
        #                                      oldval, newval, stillness,
        #                                      debug=True)
        if res == TransitionMatrix.DIFF:
            self.write_msg_diff(filehandler, res, ts, pv.name,
                                elapsed_val, update_mean, malicious_activities, score,
                                threshold, newval, oldval)

        elif res == TransitionMatrix.UNKNOWN:
            self.write_msg_unknown(filehandler, res, ts, pv.name,
                                   malicious_activities, newval, oldval)

        self.number_test_performed += 1

    def compute_transition_time(self, newval, ts, pv, filehandler, malicious_activities, transition_log):

        # Value greater (resp. lower) than max (resp. min)
        if not pv.is_bool_var() and self.is_out_of_bound(newval, self.min_val, pv):

            self.write_msg(filehandler, TransitionMatrix.UNEXPECT, ts, pv.name,
                           newval, self.min_val, malicious_activities)

        elif not pv.is_bool_var() and self.is_out_of_bound(newval, self.max_val, pv, False):
            self.write_msg(filehandler, TransitionMatrix.UNEXPECT, ts, pv.name,
                           newval, self.max_val, malicious_activities)


        # If no critical value was found, we need to compute how long the last
        # critical value remained
        found = False
        for crit_val in self.header:
            if self.find_crit_val(newval, pv) == crit_val:
                self.detection_trigger = True
                found = True
                if self.last_value is not None:
                    # The value did not change
                    if self.last_value.value == crit_val:
                        # This is the case where a loop transition occured
                        # We update the end time to compute the remaining of
                        # the critical value. The second case means that the
                        # variable left the critical value than came back so
                        # we need to update when it started

                        # check if the change was done due to the noise
                        if not self.has_changed or (ts - self.last_value.end == 1):
                            self.last_value = ValueTS(value=crit_val,
                                                      start=self.last_value.start,
                                                      end=ts)
                            self.update_step_still(newval - self.last_exact_val)
                        else:
                            self.last_value = ValueTS(value=crit_val,
                                                      start=ts,
                                                      end=ts)
                            self.update_step_still = welford.Welford()

                        # This test is performed every remaining time to detect attack
                        # faster or not because it require a lot of processing power
                        if pv.is_bool_var():
                            elapsed_time = int((ts - self.last_value.start).total_seconds())
                            pattern = self.get_pattern(self.last_value.value, self.last_value.value)
                            # 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5
                            if not isinstance(pattern, int) and pattern.has_cluster() and elapsed_time > pattern.max_time:
                                if self.update_step_still.k > 0:
                                    update_mean = self.update_step_still.mean
                                else:
                                    update_mean = newval - self.last_exact_val

                                if self.control_detection % 3 == 0:

                                    self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                                elapsed_time, update_mean, self.last_value.value,
                                                                self.last_value.value, stillness=True)
                                self.control_detection += 1

                        self.update_step = welford.Welford()

                    # The value has changed since last time
                    else:
                        self.control_detection = 0
                        self.slowing_phase = False
                        self.last_check = None
                        if not pv.is_bool_var():
                            elapsed_trans_t = int((ts - self.last_value.end).total_seconds())
                            if self.update_step.k > 0:
                                # The very last step to get to the crit val
                                self.update_step(newval - self.last_exact_val)
                                update_mean = self.update_step.mean
                            else:
                                update_mean = newval - self.last_exact_val

                            self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                        elapsed_trans_t, update_mean,
                                                        self.last_value.value,
                                                        crit_val)

                            transition_log.append(TransitionLogEntry(pv.name, self.last_value.value,
                                                                     crit_val, self.last_value.end,
                                                                     ts, elapsed_trans_t, update_mean))
                        # When a value change from one critical value to another without transition
                        # value so we must compute how long a value remained (only boolean?)
                        else:
                            same_value_t = int((self.last_value.end - self.last_value.start).total_seconds())

                            if self.update_step_still.k > 0:
                                update_mean = self.update_step_still.mean
                            else:
                                update_mean = newval - self.last_exact_val

                            self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                        same_value_t, update_mean, self.last_value.value,
                                                        self.last_value.value, stillness=True)

                            transition_log.append(TransitionLogEntry(pv.name, self.last_value.value,
                                                                     self.last_value.value, self.last_value.end,
                                                                     ts, same_value_t, update_mean))

                        self.last_value = ValueTS(value=crit_val, start=ts, end=ts)

                        self.update_step = welford.Welford()
                        self.update_step_still = welford.Welford()
                else:
                    self.last_value = ValueTS(value=crit_val, start=ts, end=ts)
                    # can happen if you are directly on a critical value when
                    # starting reading the trace
                    if self.last_exact_val is not None:
                        self.update_step(newval - self.last_exact_val)

                self.has_changed = False
                break

            if newval < crit_val:
                break

        # There is no more modification than will occur
        if not found or ts == self.end_time:
            self.has_changed = True

            if self.last_exact_val is not None:
                self.update_step(newval - self.last_exact_val)

            if self.last_value is not None:

                in_crit = self.find_in_between_crit_val(self.last_exact_val, newval, pv)
                if in_crit is not None and in_crit != self.last_value.value:
                    elapsed_trans_t = int((ts - self.last_value.end).total_seconds())

                    self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                elapsed_trans_t, self.update_step.mean,
                                                self.last_value.value, in_crit)

            # Case where a variable was on critical value and just changed to non-critical
            # We need to see how long it remains in that critical value
            if self.detection_trigger and self.last_value is not None:
                self.detection_trigger = False

            # To accelerate attack detection, perform the detection process when too long transition
            # time
            if not pv.is_bool_var() and self.last_value is not None:
                elapsed_time = int((ts - self.last_value.end).total_seconds())
                current_trend = self.update_step.mean
                target = None
                range_index, _ = pv.digitizer.get_range(newval)
                if current_trend >= 0 and self.last_value.value < self.header[-1]:
                    target = self.last_value.value + 1
                    exceed = range_index > pv.limit_values[target][0]
                elif current_trend < 0 and self.last_value.value > self.header[0]:
                    target = self.last_value.value - 1
                    exceed = range_index < pv.limit_values[target][0]

                if target is not None:
                    pattern = self.get_pattern(self.last_value.value, target)

                    if pattern.has_cluster() and elapsed_time > pattern.max_time and not exceed:
                        self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                    elapsed_time, current_trend, self.last_value.value, target)

            self.update_step_still = welford.Welford()
        self.last_exact_val = newval

    def add_update_step_same(self, value, row):
        if self.update_step_still.k != 0:
            self.transitions[row][row].add_update_step(self.update_step_still.mean)
        else:
            self.transitions[row][row].add_update_step(value - self.last_exact_val)
        self.update_step_still = welford.Welford()


    def add_update_step_diff(self, row, column, value):
        if self.update_step.k != 0:
            self.update_step(value - self.last_exact_val)
            self.transitions[row][column].add_update_step(self.update_step.mean)
        else:
            self.transitions[row][column].add_update_step(value - self.last_exact_val)
        self.update_step = welford.Welford()


    # Add the elasped times of each transition.
    @Decorators
    def update_transition_matrix(self, value, ts, pv):
        found = False
        for crit_val in self.header:
            if self.find_crit_val(value, pv) == crit_val:
                found = True

                if self.last_val_train is not None:
                    row = self.val_pos[self.last_val_train.value]
                    column = self.val_pos[crit_val]

                    if self.transitions[row][column] == -1:
                        self.transitions[row][column] = TimePattern(bool_var=pv.is_bool_var(), same_crit_val=False)

                    if self.transitions[row][row] == 0:
                        self.transitions[row][row] = TimePattern(bool_var=pv.is_bool_var(), same_crit_val=True)

                    if self.last_val_train.value == crit_val:

                        if not self.has_changed or (ts - self.last_val_train.end == 1):
                            self.last_val_train = ValueTS(value=crit_val,
                                                          start=self.last_val_train.start,
                                                          end=ts)
                            # Keeping track of update in the critical zone
                            self.update_step_still(value - self.last_exact_val)
                        else:
                            # Since it has changed, we have to compute last value

                            self.last_val_train = ValueTS(value=crit_val,
                                                          start=ts,
                                                          end=ts)

                            self.update_step_still = welford.Welford()

                        # Reset of the update list because we came back
                        self.update_step = welford.Welford()
                        self.computation_trigger = False
                    else:
                        same_value_t = int((self.last_val_train.end - self.last_val_train.start).total_seconds())
                        self.transitions[row][row].update(same_value_t)

                        # We have to store the update behavior in the range
                        self.add_update_step_same(value, row)
                        elapsed_trans_t = int((ts - self.last_val_train.end).total_seconds())

                        # The transition should be from consecutive critical value
                        # It is not true if the width of a range is really small

                        self.transitions[row][column].update(elapsed_trans_t)

                        # A new transition was completed so we have to store the update behavior
                        # If the transition is between successive range there put directly the update step
                        self.add_update_step_diff(row, column, value)

                        self.last_val_train = ValueTS(value=crit_val, start=ts, end=ts)
                        self.computation_trigger = True
                else:
                    self.last_val_train = ValueTS(value=crit_val, start=ts, end=ts)
                    if self.last_exact_val is not None:
                        self.update_step(value - self.last_exact_val)

                    self.computation_trigger = False
                self.has_changed = False
                break
            else:
                self.computation_trigger = False
                if value < crit_val:
                    break

        if not found and self.last_val_train is not None:
            self.has_changed = True

            #Since the variable left a critical value, we need to keep track of the
            #behavior of update to complete the next transition
            self.update_step(value - self.last_exact_val)

            # If the update step is too big and the width of range is to small, the IDS may miss
            # a transition
            in_crit = self.find_in_between_crit_val(self.last_exact_val, value, pv)
            if in_crit is not None and in_crit != self.last_val_train.value:
                row = self.val_pos[self.last_val_train.value]
                column = self.val_pos[in_crit]
                if self.transitions[row][column] == -1:
                    self.transitions[row][column] = TimePattern(bool_var=pv.is_bool_var(), same_crit_val=False)

                elapsed_trans_t = int((ts - self.last_val_train.end).total_seconds())
                self.transitions[row][column].update(elapsed_trans_t)
                self.transitions[row][column].add_update_step(self.update_step.mean)

            if self.computation_trigger and self.last_val_train is not None:
                self.computation_trigger = False
                same_value_t = int((self.last_value.end - self.last_value.start).total_seconds())
                
                self.add_update_step_same(value, self.val_pos[self.last_val_train.value])

            self.update_step_still = welford.Welford()
            #FIXME we should check the remaining time

        # The case where the end training period was reached and no computation
        # of transition time was done because no transition time was reach
        if ts == self.end_time and not self.computation_trigger and self.last_val_train is not None:
            row = self.val_pos[self.last_val_train.value]
            same_value_t = int((self.last_val_train.end - self.last_val_train.start).total_seconds())
            self.transitions[row][row].update(same_value_t)

        self.last_exact_val = value

    def compute_elapsed_time(self):
        elapsed_time = []
        for i in range(len(self.historic_val) - 1):
            cur = self.historic_val[i].ts
            nextframe = self.historic_val[i+1].ts
            elapsed_time.append(int((nextframe - cur).total_seconds()))
        return elapsed_time

    def compute_clusters(self):
        for row in range(len(self.header)):
            for column in range(len(self.header)):
                entry = self.transitions[row][column]
                if not isinstance(entry, int):
                    if self.strategy == LOF:
                        entry.compute_clusters(name=self.name, row=row, col=column)
                    elif self.strategy == ISO:
                        entry.train_forest(name=self.name, row=row, col=column)
                    elif self.strategy == SVM:
                        entry.train_svm(name=self.name, row=row, col=column)
                    else:
                        raise ValueError("Unknown strategy: {}".format(self.strategy))

class TimeChecker(Checker):

    def __init__(self, data, pv_store, filename, strategy=LOF, noisy=True,
                 detection_store=None):

        Checker.__init__(self, pv_store, data)
        self.done = False
        self.detection_store = detection_store
        self.messages = {}

        self.noisy = noisy

        self.map_pv_cond = {}
        self.map_var_frame = {}

        self.detection_cond = {}
        self.filehandler = open(filename, "w+")
        self.matrices = None

        self.malicious_activities = OrderedDict()
        self.elapsed_time_per_computation = list()

        self.transition_log = list()

        self.strategy = strategy

    def close(self):
        self.filehandler.close()

    def create_matrices(self):
        matrices = {}
        for name, variable in self.vars.items():
            if variable.is_periodic and not variable.ignore:
                matrices[name] = TransitionMatrix(variable, self.strategy, self.noisy)
        self.matrices = matrices

    def is_var_of_interest(self, name):
        return (name != 'timestamp' and name != 'normal/attack'
                and self.vars[name].is_periodic and not self.vars[name].ignore and len(self.vars[name].limit_values) > 0)

    def basic_variable(self, name):
        return (name != 'timestamp' and name != 'normal/attack'
                and not self.vars[name].ignore and len(self.vars[name].limit_values) == 0)

    def basic_detection(self, name, value, ts):
        pv = self.vars[name]
        if (value > pv.max_val and
                utils.normalized_dist(pv.max_val, pv.min_val, value, pv.max_val) >= utils.DIST):
            self.filehandler.write("[{}] Value too high for {} expected:{}, got:{}\n".format(ts, name,
                                                                                             pv.max_val, value))
        elif (value < pv.min_val and
              utils.normalized_dist(pv.max_val, pv.min_val, value, pv.min_val) >= utils.DIST):
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
            if val.is_periodic and not val.ignore and len(val.limit_values) > 0:
                self.matrices[name].compute_clusters()
                self.matrices[name].last_exact_val = None
                self.matrices[name].update_step = welford.Welford()

    def detect_suspect_transition(self):
        nbr_state = len(self.detection_store)
        nbr_test_perf = 0
        start_timer = timer()
        if self.detection_store is not None:
            for i, state in enumerate(self.detection_store):
                if i % 3600 == 0:
                    end_timer = timer()
                    print("Elasped time: {}".format(end_timer - start_timer))
                    self.elapsed_time_per_computation.append((end_timer - start_timer))
                    print("IDSTimechecker Starting state {} of {}, test_performed:{}".format(i, nbr_state, nbr_test_perf))
                    nbr_test_perf = 0
                    start_timer = timer()
                ts = state['timestamp']
                for name, val in state.items():
                    if self.is_var_of_interest(name):
                        matrix = self.matrices[name]
                        if i == len(self.detection_store) - 1:
                            matrix.end_time = ts
                        pv = self.vars[name]
                        matrix.number_test_performed = 0
                        matrix.compute_transition_time(val, ts, pv, self.filehandler,
                                                       self.malicious_activities,
                                                       self.transition_log)
                        try:
                            assert matrix.number_test_performed <= 1
                        except AssertionError:
                            pdb.set_trace()
                        nbr_test_perf += matrix.number_test_performed
                    elif self.basic_variable(name):
                        pass
                        #self.basic_detection(name, val, ts)
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

    def export_matrix(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.matrices, f)

    def import_matrix(self, filename):
        with open(filename, "rb") as f:
            mat = pickle.load(f)
            self.matrices = mat

    def export_transition_log(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.transition_log, f)

    def import_transition_log(self, filename):
        with open(filename, "rb") as f:
            log = pickle.load(f)
            self.transition_log = log

    def get_vars_alerts_hist(self):
        alert_occurence = [pv for variables in self.malicious_activities.values() for pv in variables]
        c = Counter(alert_occurence)
        total = sum(c.values(), 0.0)
        for key in c:
            c[key] /= total
        return c

    def export_detected_atk(self, filename):
        with open(filename,"w") as f:
            for k, v in self.malicious_activities.items():
                f.write("[{}]: {}\n".format(k, v))

