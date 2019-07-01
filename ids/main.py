#!/usr/bin/env python3
import argparse
import queue
import pickle

from reader import Reader
from pvStore import PVStore
from reqChecker import ReqChecker
from timeChecker import TimeChecker

parser = argparse.ArgumentParser()

parser.add_argument("--conf", type=str, dest="conf")
parser.add_argument("--infile", type=str, dest="infile")

def main_network(conf, infile):
    queue_req = queue.Queue()
    queue_time = queue.Queue()

    threads = []

    reader = Reader(infile, [queue_req, queue_time])
    threads.append(reader)
    reader.start()

    """
    req_checker = ReqChecker(conf, queue_req)
    threads.append(req_checker)
    req_checker.start()
    """

    time_checker = TimeChecker(conf, queue_time)
    threads.append(time_checker)
    time_checker.start()

    for thr in threads:
        thr.join()

def main(conf, infile):

    data = pickle.load(open(infile, "rb"))

    time_checker = TimeChecker(conf, data)
    time_checker.start()
    time_checker.join()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.conf, args.infile)
