#!/usr/bin/env python3
import collections
import pickle
from collections import OrderedDict, Counter
from datetime import datetime, timedelta
import pdb
import numpy as np

import utils
from reqChecker import Checker
from timePattern import TimePattern

ValueTS = collections.namedtuple('ValueTS', ['value', 'start', 'end'])


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
        self.update_step = list()
        self.update_step_still = list()


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
            return newval < extrema - (2*pv.digitizer.width)
        else:
            return newval > extrema + (2*pv.digitizer.width)


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

    def is_close_time_noisy(self, t1, cluster, thresh=EQ_NOISY_THRESH):
        if cluster.max_val == cluster.min_val:
            return t1 == cluster.mean

        if cluster.min_val <= t1 and t1 <= cluster.max_val:
            return True

        if t1 < cluster.min_val:
            dist = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                         cluster.min_val, t1)
            return dist <= thresh

        if t1 > cluster.max_val:
            dist = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                         t1, cluster.max_val)
            return dist <= thresh


    def is_big_time_noisy(self, t1, cluster, thresh=BIG_NOISY_THRESH):
        if cluster.max_val == cluster.min_val:
            return t1 == cluster.mean

        if t1 < cluster.min_val:
            diff_min = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                             cluster.min_val, t1)
            return diff_min >= thresh

        elif t1 > cluster.max_val:
            diff_max = utils.normalized_dist(cluster.max_val, cluster.min_val,
                                             t1, cluster.max_val)

            return diff_max >= thresh

    def is_small_cluster(self, pattern, cluster):
        total_pol = sum([c.k for c in pattern.clusters])
        cluster_size  = cluster.k/total_pol

        return cluster_size < SMALL_CLUSTER

    def is_unstable_cluster(self, pattern):
        nbr_small = 0
        for c in pattern.clusters:
            if self.is_small_cluster(pattern, c):
                nbr_small += 1
        return (nbr_small/len(pattern.clusters)) > UNSTABLE_CLUSTER

    def outlier_handler(self, column, row, elapsed_time, cluster):
        pattern = self.transitions[row][column]

        if (self.is_unstable_cluster(pattern) and
                not self.is_big_time(elapsed_time, cluster.mean)):
            cluster(elapsed_time)
            return TransitionMatrix.SAME, cluster
        else:
            return TransitionMatrix.DIFF, cluster


    def outlier_handler_noisy(self, column, row, elapsed_time, cluster):
        pattern = self.transitions[row][column]

        if (self.is_unstable_cluster(pattern) and
                not self.is_big_time_noisy(elapsed_time, cluster)):
            cluster(elapsed_time)
            return TransitionMatrix.SAME, cluster
        else:
            return TransitionMatrix.DIFF, cluster

    def check_transition_time(self, newval, oldval, elapsed_time, pv):
        row = self.val_pos[oldval]
        column = self.val_pos[newval]
        expected = self.transitions[row][column]
        if expected == -1 or expected == 0:
            return TransitionMatrix.UNKNOWN, expected

        if len(expected.clusters) < N_CLUSTERS:
            # Too few information about that transition to give an opinion
            # By default we accept the transition
            return TransitionMatrix.SAME, None

        cluster = self.find_cluster(expected, elapsed_time)

        ## Test with the simulator##
        if not self.noisy:
            if not self.is_small_cluster(expected, cluster):
                if self.is_close_time_noisy(elapsed_time, cluster, thresh=EQ_THRESH):
                    return TransitionMatrix.SAME, cluster
                else:
                    return TransitionMatrix.DIFF, cluster
            else:
                if self.is_close_time_noisy(elapsed_time, cluster, thresh=EQ_THRESH):
                    return TransitionMatrix.SAME, cluster
                else:
                    return self.outlier_handler(column, row, elapsed_time, cluster)
        else:

            if not self.is_small_cluster(expected, cluster):
                if self.is_close_time_noisy(elapsed_time, cluster):
                    return TransitionMatrix.SAME, cluster
                else:
                    return TransitionMatrix.DIFF, cluster
            else:
                if self.is_close_time_noisy(elapsed_time, cluster):
                    return TransitionMatrix.SAME, cluster
                else:
                    return self.outlier_handler_noisy(column, row, elapsed_time, cluster)

    def compute_remaining(self, val, current_elapsed_time):
        row = self.val_pos[val]
        column = self.val_pos[val]
        pattern = self.transitions[row][column]
        if isinstance(pattern, int) and (pattern == 0 or pattern == -1):
            return TransitionMatrix.UNKNOWN, pattern
        # Get the maximum clusters for the remaining time
        # FIXME what about the case where a value is suddenly switched?
        cluster = pattern.clusters[-1]
        if current_elapsed_time > cluster.max_val:
            if self.noisy:
                if self.is_close_time_noisy(current_elapsed_time, cluster):
                    return TransitionMatrix.SAME, cluster
                else:
                    return TransitionMatrix.DIFF, cluster
            else:
                if self.is_close_time(current_elapsed_time, cluster.max_val):
                    return TransitionMatrix.SAME, cluster
                else:
                    return TransitionMatrix.DIFF, cluster
        return TransitionMatrix.SAME, cluster
    
    def get_pattern(self, oldval, newval):
        row = self.val_pos[oldval]
        col = self.val_pos[newval]
        return self.transitions[row][col]

    def test_validity_transition(self, elapsed_time, updates, oldval, newval):
        row = self.val_pos[oldval]
        col = self.val_pos[newval]
        pattern = self.transitions[row][col]
        # Unknown transition since there is no pattern associated
        if isinstance(pattern, int) and (pattern == 0 or pattern == -1):
            return TransitionMatrix.UNKNOWN, pattern, None

        is_outlier, score = pattern.is_outlier_hdbscan(elapsed_time, updates)
        # score is none if no cluster were defined for that transition
        if is_outlier: 
            return TransitionMatrix.DIFF, score, pattern.threshold
        else:
            return TransitionMatrix.SAME, score, pattern.threshold

    def write_msg(self, filehandler, res, ts, pv_name, got, expected,
                  malicious_activities):

        filehandler.write("[{}][{}] Unexpected value for {}, expected:{}, got {}\n".format(ts, res, pv_name,
                                                                                           expected, got))
        self.add_malicious_activities(malicious_activities, ts, pv_name)


    def write_msg_diff(self, filehandler, res, ts, pv_name, elapsed_time, update_mean,
                       malicious_activities, score, thresh, crit_val, last_val):


        filehandler.write("[{}][{}]transitions for {} {}->{}, Values:[{} {}], score:{}, thresh:{}\n".format(ts, res, pv_name,
                                                                                                            last_val, crit_val,
                                                                                                            elapsed_time, update_mean,
                                                                                                            score, thresh))
        self.add_malicious_activities(malicious_activities, ts, pv_name)


    def write_msg_unknown(self, filehandler, res, ts, pv_name, malicious_activities, crit_val, last_val):

        filehandler.write("[{}][{}]transitions for {} {}->{}\n".format(ts, res, pv_name, last_val, crit_val))

        self.add_malicious_activities(malicious_activities, ts, pv_name)


    def perform_detection_test(self, filehandler, ts, pv, malicious_activities,
                               elapsed_val, update_mean, oldval, newval):
        res, score, threshold = self.test_validity_transition(elapsed_val, update_mean,
                                                              oldval,
                                                              newval)

        if res == TransitionMatrix.DIFF:
            self.write_msg_diff(filehandler, res, ts, pv.name,
                                elapsed_val, update_mean, malicious_activities, score,
                                threshold, newval, oldval)

        elif res == TransitionMatrix.UNKNOWN:
            self.write_msg_unknown(filehandler, res, ts, pv.name,
                                   malicious_activities, newval, oldval)

    def compute_transition_time(self, newval, ts, pv, filehandler, malicious_activities):
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
                            self.update_step_still.append(newval - self.last_exact_val)
                        else:
                            self.last_value = ValueTS(value=crit_val,
                                                      start=ts,
                                                      end=ts)
                            self.update_step_still = list()

                        # This test is performed every remaining time to detect attack
                        # faster or not because it require a lot of processing power
                        if pv.is_bool_var():
                            elapsed_time = int((ts - self.last_value.start).total_seconds())
                            pattern = self.get_pattern(self.last_value.value, self.last_value.value)
                            if elapsed_time > pattern.max_time:
                                if len(self.update_step_still) > 0:
                                    update_mean = np.mean(self.update_step_still)
                                else:
                                    update_mean = newval - self.last_exact_val

                                self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                            elapsed_time, update_mean, self.last_value.value, self.last_value.value)

                        self.update_step = list()

                    # The value has changed since last time
                    else:
                        if not pv.is_bool_var():
                            elapsed_trans_t = int((ts - self.last_value.end).total_seconds())
                            if len(self.update_step) > 0:
                                # The very last step to get to the crit val
                                self.update_step.append(newval - self.last_exact_val)
                                update_mean = np.mean(self.update_step)
                            else:
                                update_mean = newval - self.last_exact_val

                            self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                        elapsed_trans_t, update_mean,
                                                        self.last_value.value,
                                                        crit_val)

                        # When a value change from one critical value to another without transition
                        # value so we must compute how long a value remained (only boolean?)
                        else:
                            same_value_t = int((self.last_value.end - self.last_value.start).total_seconds())

                            if len(self.update_step_still) > 0:
                                update_mean = np.mean(self.update_step_still)
                            else:
                                update_mean = newval - self.last_exact_val

                            self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                        same_value_t, update_mean, self.last_value.value,
                                                        self.last_value.value)

                        self.last_value = ValueTS(value=crit_val, start=ts, end=ts)

                        self.update_step = list()
                        self.update_step_still = list()
                else:
                    self.last_value = ValueTS(value=crit_val, start=ts, end=ts)
                    # can happen if you are directly on a critical value when
                    # starting reading the trace
                    if self.last_exact_val is not None:
                        self.update_step.append(newval - self.last_exact_val)

                self.has_changed = False
                break

            if newval < crit_val:
                break


        # There is no more modification than will occur
        if not found or ts == self.end_time:
            self.has_changed = True

            if self.last_exact_val is not None:
                self.update_step.append(newval - self.last_exact_val)

            in_crit = self.find_in_between_crit_val(self.last_exact_val, newval, pv)
            if in_crit is not None and in_crit != self.last_value.value:
                row = self.val_pos[self.last_value.value]
                column = self.val_pos[in_crit]
                elapsed_trans_t = int((ts - self.last_value.end).total_seconds())

                self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                            elapsed_trans_t, np.mean(self.update_step),
                                            self.last_value.value, in_crit)

            # Case where a variable was on critical value and just changed to non-critical
            # We need to see how long it remains in that critical value
            if self.detection_trigger and self.last_value is not None:
                self.detection_trigger = False
                """
                same_value_t = (self.last_value.end - self.last_value.start).total_seconds()

                if len(self.update_step_still) > 0:
                    update_mean = np.mean(self.update_step_still)
                else:
                    update_mean = newval - self.last_exact_val

                    self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                            same_value_t, update_mean, self.last_value.value,
                                            self.last_value.value)
                """
            # To accelerate attack detection, perform the detection process when too long transition
            # time
            if not pv.is_bool_var():
                elapsed_time = int((ts - self.last_value.end).total_seconds())
                current_trend = np.mean(self.update_step)
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

                    if elapsed_time > pattern.max_time and not exceed:
                        self.perform_detection_test(filehandler, ts, pv, malicious_activities,
                                                    elapsed_time, current_trend, self.last_value.value, target)

            self.update_step_still = list()
        self.last_exact_val = newval

    def add_update_step_same(self, value, row):
        if len(self.update_step_still) != 0:
            self.transitions[row][row].add_update_step(np.mean(self.update_step_still))
        else:
            self.transitions[row][row].add_update_step(value - self.last_exact_val)
        self.update_step_still = list()


    def add_update_step_diff(self, row, column, value):
        if len(self.update_step) != 0:
            self.update_step.append(value - self.last_exact_val)
            self.transitions[row][column].add_update_step(np.mean(self.update_step))
        else:
            self.transitions[row][column].add_update_step(value - self.last_exact_val)
        self.update_step = list()


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
                        self.transitions[row][column] = TimePattern(same_crit_val=False)

                    if self.transitions[row][row] == 0:
                        self.transitions[row][row] = TimePattern(same_crit_val=True)

                    if self.last_val_train.value == crit_val:

                        if not self.has_changed or (ts - self.last_val_train.end == 1):
                            self.last_val_train = ValueTS(value=crit_val,
                                                          start=self.last_val_train.start,
                                                          end=ts)
                            # Keeping track of update in the critical zone
                            self.update_step_still.append(value - self.last_exact_val)
                        else:
                            # Since it has changed, we have to compute last value

                            #same_value_t = (self.last_val_train.end - self.last_val_train.start).total_seconds()
                            #self.transitions[row][row].update(same_value_t)
                            #self.add_update_step_same(value, row)

                            self.last_val_train = ValueTS(value=crit_val,
                                                          start=ts,
                                                          end=ts)

                            self.update_step_still = list()

                        # Reset of the update list because we came back
                        self.update_step = list()
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
                        self.update_step.append(value - self.last_exact_val)

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
            self.update_step.append(value - self.last_exact_val)

            # If the update step is too big and the width of range is to small, the IDS may miss
            # a transition
            in_crit = self.find_in_between_crit_val(self.last_exact_val, value, pv)
            if in_crit is not None and in_crit != self.last_val_train.value:
                row = self.val_pos[self.last_val_train.value]
                column = self.val_pos[in_crit]
                if self.transitions[row][column] == -1:
                    self.transitions[row][column] = TimePattern(same_crit_val=False)

                elapsed_trans_t = int((ts - self.last_val_train.end).total_seconds())
                self.transitions[row][column].update(elapsed_trans_t)
                self.transitions[row][column].add_update_step(np.mean(self.update_step))

            if self.computation_trigger and self.last_val_train is not None:
                self.computation_trigger = False
                same_value_t = int((self.last_value.end - self.last_value.start).total_seconds())
                
                self.add_update_step_same(value, self.val_pos[self.last_val_train.value])

            self.update_step_still = list()
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
                    entry.compute_clusters(name=self.name, row=row, col=column)

class TimeChecker(Checker):

    def __init__(self, data, pv_store, filename, noisy=True,
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

    def close(self):
        self.filehandler.close()

    def create_matrices(self):
        matrices = {}
        for name, variable in self.vars.items():
            if variable.is_periodic and not variable.ignore:
                matrices[name] = TransitionMatrix(variable, self.noisy)
        self.matrices = matrices

    def is_var_of_interest(self, name):
        return (name != 'timestamp' and name != 'normal/attack'
                and self.vars[name].is_periodic and not self.vars[name].ignore)

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
            if val.is_periodic and not val.ignore:
                self.matrices[name].compute_clusters()
                self.matrices[name].last_exact_val = None
                self.matrices[name].update_step = list()

    def detect_suspect_transition(self):
        nbr_state = len(self.detection_store)
        if self.detection_store is not None:
            for i, state in enumerate(self.detection_store):
                if i % 300 == 0:
                    print("Starting state {} of {}".format(i, nbr_state))
                ts = state['timestamp']
                for name, val in state.items():
                    if self.is_var_of_interest(name):
                        matrix = self.matrices[name]
                        if i == len(self.detection_store) - 1:
                            matrix.end_time = ts
                        pv = self.vars[name]
                        matrix.compute_transition_time(val, ts, pv, self.filehandler,
                                                       self.malicious_activities)
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

