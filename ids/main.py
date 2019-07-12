#!/usr/bin/env python3
import argparse
import queue
import pickle
import pdb

from reader import Reader
from pvStore import PVStore
from reqChecker import ReqChecker
from timeChecker import TimeChecker

parser = argparse.ArgumentParser()

parser.add_argument("--conf", type=str, dest="conf")
parser.add_argument("--benign", type=str, dest="infile")
parser.add_argument("--malicious", type=str, dest="malicious")

FIRST_WAVE = 7103

SECOND_WAVE = 12376

THIRD_WAVE = 16103

FOURTH_WAVE = 92573

FIFTH_WAVE = 103812

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

def main(conf, infile, malicious):
    with open(infile, "rb") as filename:
        data = pickle.load(filename)

    print("Read Normal mode file")
    with open(malicious, "rb") as mal_filename:
        data_mal = pickle.load(mal_filename)
    print("Read Attack mode file")
    data_mv = data_mal

    time_checker = TimeChecker(conf, data, detection_store=data_mv)
    time_checker.start()
    time_checker.join()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.conf, args.infile, args.malicious)
