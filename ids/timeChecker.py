#!/usr/bin/env python3
import sys
import yaml
import collections
import math
import threading
from datetime import datetime, timedelta

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

    def __repr__(self):
        return self.__str__()

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

    def create_var(self, host, port, kind, addr):
        pv = ProcessVariable(host, port, kind, addr, name=randomName())
        self.vars[pv.name] = pv
        self.map_key_name[pv.key()] = pv.name
        return pv

    def compute_frame(self):
        finish_run = {}
        current_cond = {}

        message_buff = {}

        while True:
            msg = self.store.get()
            self.done = isinstance(msg, str)
            if self.done:
                break

            key = msg.key()

            if key not in finish_run:
                finish_run[key] = False

            if all(finish_run.values()):
                break

            if key not in self.map_key_name:
                pv = self.create_var(msg.host, msg.port, msg.kind, msg.addr)
            else:
                name = self.map_key_name[key]
                pv = self.vars[name]

            if pv.first is None:
                pv.last_transition = msg.res_timestamp
                pv.first = msg.res_timestamp

            if key not in self.map_var_frame:
                self.map_var_frame[key] = [TimeFrame()]

            frame = self.map_var_frame[key][-1]

            pv.current_ts = msg.res_timestamp
            pv.value = msg.value

            # Start Debug
            if key not in self.messages:
                self.messages[key] = [msg.value]
            else:
                self.messages[key].append(msg.value)
            # End Debug

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
                    self.map_var_frame[key].append(new_frame)
                    finish_run[key] = True
                    del current_cond[key]
                else:
                    frame.add_val(msg.value, msg.res_timestamp)

    def update_condition(self, nbr_frame):
        for key, frames in self.map_var_frame.items():
            cond = TimeCond()
            for frame in frames[0:nbr_frame]:
                cond.add_expected_value(len(frame.vals))
                elapsed_time = []
                for i in range(len(frame.ts) - 1):
                    cur = frame.ts[i]
                    nextframe = frame.ts[i+1]
                    elapsed_time.append((nextframe - cur).total_seconds())

                print("{}->{}".format(key, elapsed_time))


    def display_message(self):
        s =""
        for k, v in self.messages.items():
            s += "{}->{}\n".format(k,v)
        return s

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
                diff = msg.res_timestamp - pv.last_transition
                pv.last_transition = msg.res_timestamp
                pv.elapsed_time_transition.append(diff.seconds * 1000 + diff.microseconds/1000)

            pv.current_ts = msg.res_timestamp
            pv.value = msg.value

            if key not in current_frame:
                begin = pv.first
                end = pv.first + self.frame_size
                cond = TimeCond()
                cond.set_limits(begin, end)
                current_frame[key] = cond
            else:
                current_cond = current_frame[key]
                cond = self.map_pv_cond[key]
                if pv.current_ts > current_cond.end:
                    value = pv.nbr_transition
                    res = cond.test_cond(value, np.array(pv.elapsed_time_transition))
                    if not res:
                        #print("Alert for {}".format(pv))
                        pass
                    finish_run[key] = True
                    pv.clear_time_value()

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
            pass
            #self.check_condition()
            break
