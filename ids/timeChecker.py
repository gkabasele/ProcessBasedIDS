#!/usr/bin/env python3
import sys
import yaml
import collections
import math
import threading

import numpy as np
from scipy import stats

from utils import ProcessVariable, randomName
from reqChecker import Checker

class TimeCond(object):

    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.expected_values = set() 
        self.avg_elapsed_val = [] 
        self.var_elapsed_val = [] 

    def add_expected_value(self, val):
        self.expected_values.add(val)

    def add_expected_avg_var(self, elapsed_time):
        if elapsed_time:
            avg = np.average(elapsed_time)
            var = np.var(elapsed_time)
            self.avg_elapsed_val.append(avg)
            self.var_elapsed_val.append(var)

    def compute_t(self, avg, var, ex_avg, n):
        if n > 0:
            t_score = (avg - ex_avg)/math.sqrt(var/n)
            df = n - 1
            alpha = 0.01
            crit_byte = stats.t.ppf(1-alpha, df=df)
            return t_score < crit_byte

    def test_cond(self, value, elapsed_time):
        res = value in self.expected_values
        if res:
            if elapsed_time: 
                avg = np.average(elapsed_time)
                var = np.var(elapsed_time)
                for ex_avg in self.avg_elapsed_val :
                    res = self.compute_t(avg, var, ex_avg, len(elapsed_time))
                    if res:
                        return res
        return res

class TimeChecker(Checker):

    def __init__(self, descFile, store, frameSize=5):
        Checker.__init__(self, descFile, store)
        self.done = False

        self.map_pv_cond = {}

    def create_var(self, host, port, kind, addr):
        pv = ProcessVariable(host, port, kind, addr, name=randomName())
        self.vars[pv.name] = pv
        self.map_key_name[pv.key()] = pv.name
        return pv

    def update_condition(self):
        seen = set()
        finish_run = {}

        while True:
            msg = self.store.get()
            self.done = type(msg) == str
            if self.done:
                break

            key = msg.key()

            if key not in finish_run:
                finish_run[key] = False

            if all(finish_run.values()):
                break

            if key not in self.map_key_name:
                pv = create_var(msg.host, msg.port, msg.kid, msg.addr,
                                msg.res_timestamp)
                print("Creating new variable {}".format(pv))
            else:
                name = self.map_key_name[key]
                pv = self.vars[name]

            if key not in seen:
                pv.last_transition = msg.res_timestamp
                pv.first = msg.res_timestamp
                seen.add(key)
                
            elif msg.value != pv.value:
                pv.nbr_transition +=1
                diff = msg.last_transition - pv.last_transition
                pv.last_transition = msg.res_timestamp
                pv.elapsed_time_transition(diff)
                
            pv.current_ts = msg.res_timestamp 
            pv.value = msg.value

            if key not in self.map_pv_cond:
                begin = pv.first 
                end = pv.first + self.frameSize
                cond = TimeCond(begin, end)
            else:
                cond = self.map_pv_cond[key]

                if pv.current_ts > cond.end:
                    cond.add_expected_value(pv.nbr_transition)
                    cond.average_elapsed_val = np.average(np.array(pv.elapsed_time_transition))
                    cond.variance_elapsed_val = np.var(np.array(pv.elapsed_time_transition))
                    finish_run[key] = True
                    pv.clear_time_value()

    #FIXME remove code duplication
    def check_condition(self):
        finish_run = {}
        seen = set()
        current_frame = {}

        while True:
            msg = self.store.get()
            self.done = type(msg) == str
            if self.done:
                break
            key = msg.key()

            if key not in finish_run:
                finish_run[key] = False

            if all(finish_run.values()):
                break

            if key not in self.map_key_name:
                print("Unknown variable {}".format(key))
                continue
            else:
                name = self.map_key_name[key]
                pv = self.vars[name]

            if key not in seen:
                pv.last_transition = msg.res_timestamp
                pv.first = msg.res_timestamp
                seen.add(key)

            elif msg.value != pv.value:
                pv.nbr_transition += 1
                diff = msg.last_transition - pv.last_transition
                pv.last_transition = msg.res_timestamp
                pv.elapsed_time_transition(diff)

            pv.current_ts = msg.res_timestamp
            pv.value = msg.value

            if key not in current_frame:
                begin = pv.first
                end = pv.first + self.frameSize
                cond = TimeCond(begin, end)
                current_frame[key] = cond
            else:
                current_cond = current_frame[key]
                cond = self.map_pv_cond[key]
                if pv.current_ts > current_cond.end:
                    value = pv.nbr_transition
                    res = cond.test_cond(value, np.array(pv.elasped_time_transition))
                    finish_run[key] = True
                    pv.clear_time_value()

    def run(self):
        i = 0
        while not self.done:
            if i < 5:
                self.update_condition()
                i += 1
            else:
                self.check_condition()
