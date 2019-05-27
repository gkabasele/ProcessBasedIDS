from datetime import timedelta, datetime
from queue import Queue

from utils import ProcessVariable
from reader import Reader

class PVStore(object):

    def __init__(self, queue):
        self.queue = queue
        self.store = {}

    def update_store(self):
        msg = self.queue.get()    
        key = msg.key()
        if key in self.store:
            self.update_variable(msg)
        else:
            self.store[key] = self.create_var(msg)

    def update_variable(self, msg):
        pv = self.store[msg.key()]
        if msg.value != pv.value:
            pv.nbr_transition += 1
            diff = msg.last_transition - pv.last_transition
            pv.last_transition = msg.res_timestamp
            pv.elapsed_time_transition.append(diff)

        pv.last_value = msg.value
        pv.value = msg.value

    def create_var(self, msg):
        pv = ProcessVariable(msg.host, msg.port, msg.kind, msg.addr)
        pv.last_value = msg.value
        pv.value = msg.value
        pv.last_transition = msg.res_timestamp
        self.store[msg.key()] = pv

    def items(self):
        return self.store.items()
