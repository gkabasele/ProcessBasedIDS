#!/usr/bin/env python3
import argparse
import queue
import pickle
import pdb

from reader import Reader
from pvStore import PVStore
from reqChecker import ReqChecker
from timeChecker import TimeChecker
import utils

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

    filename = "detection_v2.txt"
    threads = []

    reader = Reader(infile, [queue_req, queue_time])
    threads.append(reader)
    reader.start()

    """
    req_checker = ReqChecker(conf, queue_req)
    threads.append(req_checker)
    req_checker.start()
    """

    time_checker = TimeChecker(conf, filename, queue_time)
    threads.append(time_checker)
    time_checker.start()

    for thr in threads:
        thr.join()

def main(conf, infile, malicious):
    
    filename = "detection_v3.txt"
    print("Read Normal mode file")
    data = utils.read_state_file(infile)[:50000]
    time_checker = TimeChecker(conf, filename, data)
    time_checker.fill_matrices()
    pdb.set_trace()

    print("Read Attack mode file")
    #data_mal = utils.read_state_file(malicious)
    data_mal = utils.read_state_file(infile)[50000:100000]
    data_mv = data_mal

    time_checker.detection_store = data_mv
    time_checker.detect_suspect_transition()
    time_checker.close()
    #time_checker.start()
    #time_checker.join()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.conf, args.infile, args.malicious)
