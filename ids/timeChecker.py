#!/usr/bin/env python3
import sys
import yaml
import collections
import math
import threading
from datetime import datetime, timedelta
import pdb

import numpy as np
from scipy import stats

from utils import ProcessVariable, randomName
from reqChecker import Checker

class TimeFrame(object):

    def __init__(self):
        self.vals = []
        self.ts = []

    def add_val(self, val, ts):
        if len(self.vals) == 0 or self.vals[-1] != val:
            self.vals.append(val)
            self.ts.append(ts)

    def __str__(self):
        s = "["
        for val, ts in zip(self.vals, self.ts):
            s += " ({},{}) ".format(val, ts)

        s += "]"
        return s

    def compute_elapsed_time(self):
        elapsed_time = []
        for i in range(len(self.ts) - 1):
            cur = self.ts[i]
            nextframe = self.ts[i+1]
            elapsed_time.append((nextframe - cur).total_seconds())
        return elapsed_time

    def __repr__(self):
        s = "Val: {}, Avg:{}".format(len(self.vals),
                                     np.average(self.compute_elapsed_time()))
        return s

class TimeCond(object):

    def __init__(self):
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
        else:
            self.avg_elapsed_val.append(-1)
            self.var_elapsed_val.append(-1)

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
            if len(elapsed_time) != 0:
                avg = np.average(elapsed_time)
                var = np.var(elapsed_time)
                for ex_avg in self.avg_elapsed_val:
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

    def __init__(self, descFile, store, frameSize=10):
        Checker.__init__(self, descFile, store)
        self.done = False
        self.frame_size = timedelta(seconds=frameSize)
        self.messages = {}

        self.map_pv_cond = {}
        self.map_var_frame = {}

        self.detection_cond = {}

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

    def get_frame(self, key, detection):
        if not detection:
            if key not in self.map_var_frame:
                self.map_var_frame[key] = [TimeFrame()]
            return self.map_var_frame[key][-1]
        else:
            if key not in self.detection_cond:
                self.detection_cond[key] = [TimeFrame()]
            return self.detection_cond[key][-1]

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

            frame = self.get_frame(key, detection)

            pv.current_ts = msg.res_timestamp
            pv.value = msg.value

            if key not in current_cond and not finish_run[key]:
                begin = pv.first
                end = pv.first + self.frame_size
                cond = (begin, end)
                current_cond[key] = cond
                frame.add_val(msg.value, msg.res_timestamp)

            elif finish_run[key]:
                frame.add_val(msg.value, msg.res_timestamp)

            elif key in current_cond:
                begin, end = current_cond[key]
                if pv.current_ts > end:
                    new_frame = TimeFrame()
                    new_frame.add_val(msg.value, msg.res_timestamp)
                    if not detection:
                        self.map_var_frame[key].append(new_frame)
                    else:
                        self.detection_cond[key].append(new_frame)
                    pv.first = msg.res_timestamp
                    pv.last_transition = msg.res_timestamp
                    finish_run[key] = True
                    del current_cond[key]
                else:
                    frame.add_val(msg.value, msg.res_timestamp)

            if all(finish_run.values()):
                break

    def update_condition(self, nbr_frame):
        for key, frames in self.map_var_frame.items():
            cond = TimeCond()
            for frame in frames[0:nbr_frame]:
                cond.add_expected_value(len(frame.vals))
                elapsed_time = frame.compute_elapsed_time()
                cond.add_expected_avg_var(elapsed_time)
            self.map_pv_cond[key] = cond

    def check_condition(self):
        for key, frames in self.detection_cond.items():
            if key not in self.map_pv_cond:
                print("Alert for unknown PV {}".format(key))
                continue

            cond = self.map_pv_cond[key]
            frame = frames.pop(0)
            value = len(frame.vals)
            elapsed_time = frame.compute_elapsed_time()
            res = cond.test_cond(value, elapsed_time)
            if not res:
                print("Alert for {}\n".format(key))
                print("Got: {}\n".format(repr(frame)))
                print("Expected: {}\n".format(cond))

    def display_message(self):
        s =""
        for k, v in self.messages.items():
            s += "{}->{}\n".format(k,v)
        return s

    def display_dict(self, d):
        s = ""
        for k, v in d.items():
            s += "{}->{}\n\n".format(k, v)
        return s

    def run(self):
        i = 0
        nbr_frame = 5
        print("Creating condition")
        while not self.done and i < nbr_frame:
            self.compute_frame()
            i += 1

        self.update_condition(nbr_frame)

        print("Running detection")
        while not self.done:
            self.compute_frame(True)
            self.check_condition()
            break
