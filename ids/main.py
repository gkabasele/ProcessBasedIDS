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

def main(conf, infile, malicious, detect_file, cool, stop):

    filename = detect_file
    print("Read Normal mode file")
    if cool != 0:
        data = utils.read_state_file(infile)[cool:]
    else:
        data = utils.read_state_file(infile)
    time_checker = TimeChecker(conf, filename, data)
    time_checker.fill_matrices()
    pdb.set_trace()

    print("Read Attack mode file")
    if stop is not None:
        data_mal = utils.read_state_file(malicious)[:stop]
    else:
        data_mal = utils.read_state_file(malicious)
    data_mv = data_mal

    time_checker.detection_store = data_mv
    time_checker.detect_suspect_transition()
    time_checker.close()
    #time_checker.start()
    #time_checker.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, dest="conf")
    parser.add_argument("--benign", type=str, dest="infile")
    parser.add_argument("--malicious", type=str, dest="malicious")
    parser.add_argument("--stop", type=int, dest="stop", help="where to stop in the attack file")
    parser.add_argument("--detection", dest="detect_file", default="detection.txt")
    parser.add_argument("--cool", type=int, dest="cool", default=utils.COOL_TIME, help="size of cool time")

    args = parser.parse_args()
    main(args.conf, args.infile, args.malicious, args.detect_file,
         args.cool, args.stop)
