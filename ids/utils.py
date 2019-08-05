import struct
import sys
import random
import string
import yaml
import pickle
import math
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scapy.all import *

# PORT
MODBUS_PORT = [5020, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5029,
               5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5038, 5039,
               5040]

# TAG
SRTAG_REDIRECT = 0
SRTAG_CLONE = 1

# TCP Flags

FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80

# Variable Type

DIS_COIL = "co"
DIS_INP = "di"
HOL_REG = "hr"
INP_REG = "ir"

DISCRETE = [DIS_COIL, DIS_INP]
CONTINOUS = [INP_REG, HOL_REG]
# Funcode write

WRITE_FUNCODE = [5, 6]

TS = "timestamp"

DIST = 0.01
DIFF = 0.05
DAY_IN_SEC = 86400
COOL_TIME = 11000

class ProcessSWaTVar():

    def __init__(self, name, kind, min_val=None, max_val=None, 
                 limit_values=None):
        self.name = name
        self.kind = kind
        self.value = None
        self.first = None
        self.nbr_transition = 0
        self.last_transition = None
        self.elapsed_time_transition = []

        self.is_periodic = kind in DISCRETE

        if min_val is None:
            self.min_val = 1
        else:
            self.min_val = min_val

        if max_val is None:
            self.max_val = 2
        else:
            self.max_val = max_val

        if limit_values is None:
            self.limit_values = []
        else:
            self.limit_values = limit_values

    def __hash__(self):
        return hash(self.name)

    def is_bool_var(self):
        return self.kind in [DIS_COIL, DIS_INP]

    def clear_time_value(self):
        self.first = None
        self.nbr_transition = 0
        self.elapsed_time_transition = []

    def normalized_dist(self, val1, val2):
        return (math.sqrt((val1-val2)**2)/math.sqrt((self.max_val - self.min_val)**2))

class ProcessVariable():

    def __init__(self, host, port, kind, addr, limit_values=None, gap=1,
                 size=None, name=None, first=None):
        self.host = host
        self.port = port
        self.kind = kind
        self.addr = addr
        self.name = name
        self.size = size
        self.gap = gap
        self.first = None
        self.nbr_transition = 0
        self.last_transition = None
        self.elapsed_time_transition = []
        self.current_ts = None
        self.value = None
        if limit_values:
            self.limit_values = limit_values
        else:
            self.limit_values = []

    @classmethod
    def funcode_to_kind(cls, funcode):
        if funcode in [1, 5, 15]:
            return DIS_COIL
        elif funcode == 2:
            return DIS_INP
        elif funcode in [3, 6, 10, 22, 23]:
            return HOL_REG
        elif funcode == 4:
            return INP_REG

    def key(self):
        return (self.host, self.port, self.kind, self.addr)

    def __eq__(self, other):
        return ((self.host, self.port, self.kind, self.addr) ==
                (other.host, other.port, other.kind, other.addr))

    def __hash__(self):
        return hash((self.host, self.port, self.kind, self.addr))

    def __str__(self):
        return "%s : (ip: %s, port: %s, type: %s, addr: %s)" % (
                                self.name, self.host, self.port, self.kind, self.addr) 
    def __repr__(self):
        return "%s : (ip: %s, port: %s, type: %s, addr: %s)" % (
                                self.name, self.host,self.port, self.kind, self.addr) 

    def is_bool_var(self):
        return self.kind in [DIS_COIL, DIS_INP]

    def clear_time_value(self):
        self.first = None
        self.nbr_transition = 0
        self.elapsed_time_transition = []

def setup(filename, pv_store):
    with open(filename) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for var_desc in desc['variables']:
            var = var_desc['variable']
            if var['type'] == DIS_COIL or var['type'] == DIS_INP:
                limit_values = [1, 2]
                min_val = 1
                max_val = 2
            else:
                limit_values = var['values']
                min_val = var['min']
                max_val = var['max']

            pv = ProcessSWaTVar(var['name'], var['type'],
                                limit_values=limit_values,
                                min_val=min_val,
                                max_val=max_val)
            pv_store[pv.name] = pv

def randomName(stringLength=4):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

class SRTag(Packet):
    name = "SRTag"
    fields_desc = [IPField("dst", None),
                   ShortField("identifier", None),
                   ByteField("protocol", None),
                   ByteField("reason", None)
                  ]

class ModbusReq(Packet):
    name = "ModbusReq"
    fields_desc = [ShortField("transId", 0),
                   ShortField("protoID", 0),
                   ShortField("length", None),
                   ByteField("unitID", 0),
                   ByteField("funcode", None),
                   ShortField("startAddr", 0)
                  ]

class ModbusRes(Packet):
    name = "ModbusRes"
    fields_desc = [ShortField("transId", 0),
                   ShortField("protoID", 0),
                   ShortField("length", None),
                   ByteField("unitID", 0),
                   ByteField("funcode", None)
                  ]

for port in MODBUS_PORT:
    bind_layers(TCP, ModbusReq, dport=port)
    bind_layers(TCP, ModbusRes, sport=port)

class ReadCoilsRes(Packet):
    name = "ReadCoilsRes"
    fields_desc = [BitFieldLenField("count", None, 8, count_of="status"),
                   FieldListField("status", [0x00], ByteField("", 0x00), count_from=lambda x:x.count)
                  ]
bind_layers(ModbusRes, ReadCoilsRes, funcode=1)

class ReadDiscreteRes(Packet):
    name = "ReadDiscreteRes"
    fields_desc = [BitFieldLenField("count", None, 8, count_of="status"),
                   FieldListField("status", [0x00], ByteField("", 0x00), count_from=lambda x:x.count)
                  ]
bind_layers(ModbusRes, ReadDiscreteRes, funcode=2)

class ReadHoldRegRes(Packet):
    name = "ReadHoldRegRes"
    fields_desc = [BitFieldLenField("count", None, 8, count_of="value", adjust=lambda pkt, x: x*2),
                   FieldListField("value", [0x0000], ShortField("", 0x0000), count_from=lambda x: x.count)
                  ]
bind_layers(ModbusRes, ReadHoldRegRes, funcode=3)

class ReadInputRes(Packet):
    name = "ReadInputRes"
    fields_desc = [BitFieldLenField("count", None, 8, count_of="registers", adjust=lambda pkt, x: x*2),
                   FieldListField("registers", [0x0000], ShortField("", 0x0000), count_from=lambda x:x.count)
                  ]
bind_layers(ModbusRes, ReadInputRes, funcode=4)

class WriteSingleCoilRes(Packet):
    name = "WriteSingleCoilRes"
    fields_desc = [ShortField("addr",None),
                   ShortField("value",None)
                  ]
bind_layers(ModbusRes, WriteSingleCoilRes, funcode=5)

class WriteSingleRegRes(Packet):
    name = "WriteSingleRegRes"
    fields_desc = [ShortField("addr", None),
                   ShortField("value", None)
                  ]
bind_layers(ModbusRes, WriteSingleRegRes, funcode=6)


# Translation between funcode and field name
func_fields_dict = {
                     1 : "status",
                     2 : "status",
                     3 : "value", 
                     4 : "registers",
                     5 : "value",
                     6 : "value",
                   }

def is_number(s):
    """ Returns Truse if string s is a number """
    return s.replace('.','',1).isdigit()

def same_value(max_val, min_val, val1, val2, thresh=DIST):
    return normalized_dist(max_val, min_val, val1, val2) <= thresh

def normalized_dist(max_val, min_val, val1, val2):
    return (math.sqrt((val1-val2)**2)/math.sqrt((max_val - min_val)**2))

def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid')/window

def show_kde(data, name=None):
    xs, y_data = compute_kde(data) 
    n, bins, patches = plt.hist(data, 150, density=True)
    if name is not None:
        plt.title(name)
    plt.plot(xs, y_data)
    plt.show()

def compute_kde(data):
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 150)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    y_data = density(xs)
    return xs, y_data


def read_state_file(name):
    with open(name, "rb") as filename:
        data = pickle.load(filename)
    return data

def get_all_values_pv(data, pvname, limit=None):
    if limit is None:
        return np.array([x[pvname] for x in data])
    else:
        return np.array([x[pvname] for x in data[:limit]])
