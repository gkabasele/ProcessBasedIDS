#!/usr/bin/env python3
import sys
import yaml
import collections
import math

import numpy as np
from scipy import stats

from utils import ProcessVariable
from reqChecker import Checker

class TimeCond(object):

    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.expected_values = set() 
        self.avg_elapsed_val = 0 
        self.var_elapsed_val = 0

    def add_expected_value(self, val):
        self.expected_values.add(val)

    def test_cond(self, value, elapsed_time):
        res = value in self.expected_values
        if res:
            avg = np.average(elapsed_time)
            var = np.var(elapsed_time)
            t_score = (avg - self.avg_elapsed_val)/math.sqrt(var/len(elapsed_time))

            df = len(elapsed_time) - 1
            alpha = 0.01
            crit_byte = stats.t.ppf(1-alpha, df=df)

            res = t_score < crit_byte
        return res

def TimeChecker(Checker):

    def __init__(self, descFile, store, frameSize=5):
        Checker.__init__(self, descFile, store)

        self.map_pv_cond = {}

    def update_condition(self):
        for k, v in self.store.items():
            if k not in self.map_pv_cond:
                begin = v.first 
                end = v.first + self.frameSize
                cond = TimeCond(begin, end)
                self.map_pv_cond[k] = cond
            else:
                cond = self.map_pv_cond[k]
                if v.current_ts > cond.end:
                    cond.add_expected_value(v.nbr_transition)
                    cond.average_elapsed_val = np.average(np.array(v.elasped_time_transition))
                    cond.variance_elapsed_val = np.var(np.array(v.elasped_time_transition))
                    v.clear_time()

    def check_condition(self):
        current_frame = {}
        for k, v in self.store.items():
            if k not in current_frame:
                begin = v.first
                end = v.first + self.frameSize
                cond = TimeCond(begin, end)
                current_frame[k] = cond
            else:
                current_cond = current_frame[k]
                cond = self.map_pv_cond[k]
                if v.current_ts > current_cond.end:
                    value = v.nbr_transition
                    res = cond.test_cond(value, np.array(v.elasped_time_transition))
    def run(self):
        pass
