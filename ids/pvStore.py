from datetime import timedelta, datetime
from queue import Queue
import copy

from utils import ProcessVariable
from reader import Reader

class PVStore(object):

    def __init__(self, queue):
        self.queue = queue
        self.store = [] 
        self.pv_indices = {}
        self.end = False

    def update_store(self):
        msg = self.queue.get()    
        self.end = type(msg) == str
        if not self.end:
            key = msg.key()
            if key in self.pv_indices:
                self.update_variable(key, msg)
            else:
                self.store.append(self.create_var(msg))
                self.pv_indices[key] = len(self.store) - 1

    def update_variable(self, key, msg):
        pv = copy.deepcopy(self.store[self.pv_indices[key]])
        if msg.value != pv.value:
            pv.nbr_transition += 1
            diff = msg.last_transition - pv.last_transition
            pv.last_transition = msg.res_timestamp
            pv.elapsed_time_transition.append(diff)

        if pv.first is None:
            pv.first = msg.res_timestamp

        pv.last_ts = msg.res_timestamp
        pv.value = msg.value

        self.store.append(pv)
        self.pv_indices[key] = len(self.store) - 1

    def create_var(self, msg):
        pv = ProcessVariable(msg.host, msg.port, msg.kind, msg.addr)
        pv.value = msg.value
        pv.last_transition = msg.res_timestamp
        pv.first = msg.res_timestamp
        self.store[msg.key()] = pv

    def items(self):
        return self.store.items()
