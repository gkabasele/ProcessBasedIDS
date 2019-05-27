from queue import Queue

from util import *

class PVStore(object):

    def __init__(self, queue):
        self.queue = queue
        self.store = {}

    def update_store(self):
        pass
