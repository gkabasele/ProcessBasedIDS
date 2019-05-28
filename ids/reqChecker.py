#!/usr/bin/env python3

import sys
import os
import argparse
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
        return len(filter(lambda x: x.is_bool_var(), self.vars.values()))

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
            req = Requirement(Parser(Lexer(req_desc['requirement'])).parse())
            self.reqs.append(req)

    def update_vars_from_store(self):
        seen = set()

        while True:
            msg = self.store.get()
            self.done = type(msg) == str
            if self.done:
                break
            key = msg.key()

            if key in seen:
                break

            if key in self.map_key_name:
                name = self.map_key_name[key]
                self.vars[name].value = msg.value
                seen.add(key)
            else:
                print("Unknown ProcessVariable {}".format(key))
                seen.add(key)

    def run(self):
        while not self.done:
            self.update_vars_from_store()
            self.get_min_distance()
