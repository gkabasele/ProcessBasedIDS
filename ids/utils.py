import struct
import sys
import random
import string
import math
import numpy as np
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

# Funcode write

WRITE_FUNCODE = [5, 6]

TS = "timestamp"

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

        if min_val:
            self.min_val = min_val
        else:
            self.min_val = 1

        if max_val:
            self.max_val = max_val
        else:
            self.max_val = 2

        if limit_values:
            self.limit_values = limit_values
        else:
            self.limit_values = []

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
