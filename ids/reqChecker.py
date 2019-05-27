#!/usr/bin/env python3

import sys
import argparse
import yaml
import collections

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../requirement_interpreter')))

from utils import ProcessVariable

import Lexer
import Parser
import Interpreter

class Requirement(object):

    identifier = 0

    def __init__(self, content):

        self.identifier = Requirement.identifier
        Requirement.identifier += 1
        self.content = content

class State(object):

    def __init__(self, descFile, store, bool_weight=5, num_weight=1):
        #name -> Process Variable
        self.vars = {}
        #key -> name
        self.map_key_name = {}
        self.reqs = [] 

        self.bool_weight = bool_weight
        self.num_weight = num_weight

        self.store = store

        self.setup(descFile)

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

    def setup(self, descFile):
        content = open(descFile).read()
        desc = yaml.load(content)
        for var_desc in desc['variables']:
            var = var_desc['variable']
            pv = ProcessVariable(var['host'], var['port'], var['type'],
                                 var['address'], var.get('gap', 1), var['size'],
                                 var['name'])
            self.vars[pv.name] = pv
            self.map_key_name[pv.key()] = pv.name

        for req_desc in desc['requirements']:
            req = Requirement(Parser(Lexer(req_desc['requirement'])).parse())
            self.reqs.append(req)

    def update_vars_from_store(self):
        for k, v in self.store.items():
            val = v.value
            if k in self.map_key_name:
                name = self.map_key_name[k]
                self.vars[name].value = val
            else:
                print("Unknown Process Variable {}".format(v))
