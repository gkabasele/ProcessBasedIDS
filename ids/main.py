#!/usr/bin/env python3
import argparse
from queue import Queue

from reader import Reader
from pvStore import PVStore
from reqChecker import ReqChecker
from timeChecker import TimeChecker

parser = argparse.ArgumentParser()

parser.add_argument("--conf", type=str, dest="conf")
parser.add_argument("--infile", type=str, dest="infile")

def main(conf, infile):
    queue = Queue()

    reader = Reader(infile, queue)
    store = PVStore(queue)
    req_checker = ReqChecker(conf, store)
    time_checker = TimeChecker(conf, store)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.conf, args.infile)
