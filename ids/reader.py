#!/usr/bin/env python3

import argparse
import queue 
import threading

import time
from datetime import datetime

from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP
from utils import *

class PVMessage(object):

    __slots__ = ['host', 'port', 'addr', 'size',
                 'value', 'kind', 'req_timestamp', 'res_timestamp']

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

class Reader(threading.Thread):

    def __init__(self, trace, queues):
        threading.Thread.__init__(self)
        self.map_req_res = {}
        self.trace = trace
        self.queues = queues

    def get_ip_tcp_fields(self, ip_pkt, tcp_pkt):
        srcip = ip_pkt.src
        dstip = ip_pkt.dst
        sport = tcp_pkt.sport
        dport = tcp_pkt.dport
        return srcip, dstip, sport, dport

    def get_modbus_req_fields(self, pkt):
        funcode = pkt.funcode
        transId = pkt.transId
        addr = pkt.startAddr
        kind = ProcessVariable.funcode_to_kind(funcode)
        return funcode, transId, addr, kind

    def get_variable_val(self, funcode, payload):
        val = payload.getfieldval(func_fields_dict[funcode])
        if type(val) is list:
            val = val[0]
        return val

    def get_pkt_time(self, pkt):
        ts_subsec = pkt.usec
        ts_sec_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pkt.sec))
        ts_str = '{}.{}'.format(ts_sec_str, ts_subsec)
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")

    def read_packet(self, pkt, meta_pkt):
        ether_pkt = Ether(pkt)
        if ether_pkt.type == 0x0800:
            ip_pkt = ether_pkt[IP]
            if ip_pkt.proto == 6:
                tcp_pkt = ip_pkt[TCP]
                srcip, dstip, sport, dport = self.get_ip_tcp_fields(ip_pkt, tcp_pkt)
                flags = tcp_pkt.flags
                if PSH & flags:
                    if dport == MODBUS_PORT:
                        modbus_pkt = tcp_pkt[ModbusReq]
                        funcode, transId, addr, kind = self.get_modbus_req_fields(modbus_pkt)
                        self.map_req_res[(transId, dstip)] = PVMessage(dstip, kind, addr,
                                                                       self.get_pkt_time(meta_pkt))
                    elif sport == MODBUS_PORT:
                        modbus_pkt = tcp_pkt[ModbusRes]
                        transId = modbus_pkt.transId
                        msg = self.map_req_res[(transId, srcip)]
                        msg.value = self.get_variable_val(modbus_pkt.funcode,
                                                          modbus_pkt.payload)
                        msg.res_timestamp = self.get_pkt_time(meta_pkt)
                        for q in self.queues:
                            q.put(msg)
                        self.map_req_res.pop((srcip, transId), None)

    def run(self):
        for pkt, meta_pkt in RawPcapReader(self.trace):
            self.read_packet(pkt, meta_pkt)
        for q in self.queues:
            q.put("End")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--trace", type=str, dest="trace")
    args = parser.parse_args()

    queues = [queue.Queue()]
    reader = Reader(args.trace, queues)
    reader.start()
    reader.join()
