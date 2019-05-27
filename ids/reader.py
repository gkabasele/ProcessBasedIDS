#!/usr/bin/env python3

import argparse
from queue import Queue

from scapy.all import *
from util import *

class Message(object):

    __slots__ = ['host', 'port', 'addr', 'size', 'value']

    def __init__(self, host, kind, addr, ts, port=MODBUS_PORT):
        self.req_timestamp = ts
        self.res_timestamp = None
        self.host = host
        self.port = port
        self.kind = kind
        self.addr = addr
        self.value = None

    def key(self):
        return (self.host, self.port, self.kind, self.addr)

class Reader(object):

    def __init__(self, trace, queue):
        self.map_req_res = {}
        self.trace = trace
        self.reader = PcapReader(trace)
        self.queue = queue

    def get_ip_tcp_fields(self, pkt):
        srcip = pkt[IP].src
        dstip = pkt[IP].dst
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
        return srcip, dstip, sport, dport
    
    def get_modbus_req_fields(self, pkt):
        funcode = pkt[ModbusReq].funcode
        transId = pkt[ModbusReq].transId
        addr = pkt[ModbusReq].startAddr
        kind = ProcessVariable.funcode_to_kind(funcode)
        return funcode, transId, addr, kind

    def get_variable_val(self, funcode, payload):
        val = payload.getfieldval(func_fields_dict[funcode])
        if type(val) is list:
            val = val[0]
        return val

    def read_packet(self, packet):
        payload = packet.get_payload()
        pkt = IP(payload)
        if TCP in pkt:
            srcip, dstip, sport, dport = self.get_ip_tcp_fields(pkt)
            if dport == MODBUS_PORT:
                funcode, transId, addr, kind = self.get_modbus_req_fields(pkt)
                self.map_req_res[(transId, dstip)] = Message(dstip, kind, addr,
                                                             packet.time)
            elif sport == MODBUS_PORT:
                transId = pkt[ModbusRes].transId
                msg = self.map_req_res[(srcip, transId)] 
                msg.value = self.get_variable_val(pkt[ModbusRes].funcode,
                                                  pkt[ModbusRes].payload)
                msg.res_timestamp = packet.time
                self.queue.put(msg)
                self.map_req_res.pop((srcip, transId), None)

    def readall(self):
        for p in self.reader:
            self.read_packet(p)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--trace", type=str, dest="trace")
    args = parser.parse_args()

    queue = Queue()
    reader = Reader(args.trace)
