#!/usr/bin/env python3

import sys
import os
import yaml
import collections
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../requirement_interpreter')))

from utils import ProcessVariable

from lexer import Lexer
from parser import Parser
from interpreter import Interpreter 

class Requirement(object):

    identifier = 0

    def __init__(self, content):

        self.identifier = Requirement.identifier
        Requirement.identifier += 1
        self.content = content

    def __str__(self):
        return "ID:{}, content:{}".format(self.identifier, self.content)

    def __repr__(self):
        return self.__str__()

class Checker(threading.Thread):

    def __init__(self, descFile, queue):
        threading.Thread.__init__(self)
        # name -> Process Variable
        self.vars = {}
        #key -> name
        self.map_key_name = {}

        self.store = queue
        self.setup(descFile)

    def setup(self, descFile):
        content = open(descFile).read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for var_desc in desc['variables']:
            var = var_desc['variable']
            pv = ProcessVariable(var['host'], var['port'], var['type'],
                                 var['address'], var.get('gap', 1),
                                 size=var['size'], name=var['name'])

            self.vars[pv.name] = pv
            self.map_key_name[pv.key()] = pv.name

    def run(self):
        raise NotImplementedError

class ReqChecker(Checker):

    def __init__(self, descFile, store, bool_weight=5, num_weight=1):

        Checker.__init__(self, descFile, store)
        self.reqs = [] 
        self.bool_weight = bool_weight
        self.num_weight = num_weight
        self.done = False

        self.create_requirement(descFile)

    def count_bool_var(self):
        return len([x for x in filter(lambda x: x.is_bool_var(), self.vars.values())])

    def get_min_distance(self):

        min_dist = None
        identifier = None
        bool_var = self.count_bool_var()
        num_var = len(self.vars) - bool_var

        for req in self.reqs:
            tmp = min_dist
            i = Interpreter(None, self.vars, self.num_weight, self.bool_weight)
            violation = i.visit(req.content)
            if violation:
                pass
            if min_dist is None:
                min_dist = i.compute_distance(num_var, bool_var)
                identifier = req.identifier
            else:
                d = i.compute_distance(num_var, bool_var)
                min_dist = min(min_dist, d)
                identifier = req.identifier if tmp != min_dist else identifier
        Distance = collections.namedtuple('Distance', 
                                          ['max_dist', 'max_identifier',
                                           'min_dist', 'min_identifier'])
        d = Distance(min_dist, identifier, min_dist, identifier)
        return d

    def create_requirement(self, descFile):
        content = open(descFile).read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for req_desc in desc['requirements']:
            content = req_desc['requirement'].replace("\n","")
            req = Requirement(Parser(Lexer(req_desc['requirement'])).parse())
            self.reqs.append(req)

    def update_vars_from_store(self):
        seen = {}
        message_read = 0
        for k in self.vars.keys():
            seen[k] = False

        while True:

            if all(seen.values()):
                break

            msg = self.store.get()
            message_read += 1
            self.done = isinstance(msg, str)
            if self.done:
                break
            key = msg.key()

            if key in self.map_key_name:
                name = self.map_key_name[key]
                pv = self.vars[name]
                pv.value = msg.value
                print("Updating PV: {}".format(pv))
                seen[name] = True
            else:
                pass
                #print("Unknown ProcessVariable {}".format(key))
        return message_read

    def display_vars(self):
        s = ""
        for pv in self.vars.values():
            s += "{} ,val:{}\n".format(pv, pv.value)
        return s

    def run(self):
        all_read = 0
        while not self.done:
            all_read += self.update_vars_from_store()
            print("End: {}, Nbr Read: {}".format(self.done, all_read))
            if self.done:
                break
            self.get_min_distance()
            break
