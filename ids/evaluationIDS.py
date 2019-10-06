import argparse
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pdb
import yaml
from timeChecker import TimeChecker
import utils

MAL_CACHE = "malicious_activities.bin"
MATRIX = "matrices.bin"
LOG = "res.log"

START = 'start'
END = 'end'
TS = 'timestamp'

class EvalResult(object):

    def __init__(self, tp, fp, tn, fn, nbr_attack, missed_atk, false_atk):

        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.nbr_attack = nbr_attack
        self.missed_atk = missed_atk
        self.false_atk = false_atk

    def fpr(self):
        return self.fp/(self.tp + self.fp)

    def tpr(self):
        return self.tp/(self.tp + self.fn)

def compare_activities(mal_expected, mal_computed, attack_store):
    i = 0
    j = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    nbr_attack = 0

    attack_missed = set()
    attack_false = set()

    detect_timestamps = sorted(list(mal_computed.keys()))

    for state in attack_store:
        if i < len(mal_expected):
            attack = mal_expected[i]
        if j < len(detect_timestamps):
            detect = detect_timestamps[j]

        ts = state[TS]
        key = (ts.year, ts.month, ts.day, ts.hour, ts.minute)

        attack_detected = ts == detect
        if attack_detected:
            j += 1

        # not in an attack phase
        if ts < attack[START]:
            if attack_detected:
                false_positive += 1
                attack_false.add(key)
            else:
                true_negative += 1

        # in an attack phase
        elif ts >= attack[START] and ts < attack[END]:
            nbr_attack += 1
            if attack_detected:
                true_positive += 1
            else:
                false_negative += 1
                attack_missed.add(key)

        # end of attack phase
        elif ts == attack[END]:
            nbr_attack += 1
            if attack_detected:
                true_positive += 1
            else:
                false_negative += 1
                attack_missed.add(key)
            i += 1

        # When last attack was done
        elif ts > attack[END]:
            if attack_detected:
                false_positive += 1
                attack_false.add(key)
            else:
                true_negative += 1

    res = EvalResult(true_positive, false_positive, true_negative,
                     false_negative, nbr_attack, attack_missed,
                     attack_false)

    return res 

def create_expected_malicious_activities(atk_period):
    with open(atk_period) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        for attack in desc:
            starttime = datetime.strptime(attack[START], '%d/%m/%Y %H:%M:%S')
            endtime = datetime.strptime(attack[END], '%d/%m/%Y %H:%M:%S')
            attack[START] = starttime
            attack[END] = endtime
    return desc

def main(atk_period, conf, malicious, infile, cache):

    expected_atk = create_expected_malicious_activities(atk_period)
    data_mal = utils.read_state_file(malicious)
    if not cache:
        data = utils.read_state_file(infile)
        time_checker = TimeChecker(conf, LOG, data)
        time_checker.fill_matrices()
        pickle.dump(time_checker.matrices, open(MATRIX, "wb"))
        time_checker.detection_store = data_mal
        time_checker.detect_suspect_transition()
        pickle.dump(time_checker.malicious_activities, open(MAL_CACHE, "wb"))
        time_checker.close()
        malicious_activities = time_checker.malicious_activities
        time_checker.create_matrices()
    else:
        matrices = pickle.load(open(MATRIX, "rb"))
        malicious_activities = pickle.load(open(MAL_CACHE, "rb"))

        res = compare_activities(expected_atk, malicious_activities, data_mal)

    with open("res_eval.txt", "w") as fh:
        fh.write("Total:{}\n".format(len(data_mal)))
        fh.write("Nbr Attack: {}\n".format(res.nbr_attack))
        fh.write("TP:{}\n".format(res.tp))
        fh.write("FP:{}\n".format(res.fp))
        fh.write("TN:{}\n".format(res.tn))
        fh.write("FN:{}\n".format(res.fn))

        tpr = res.tpr()
        fpr = res.fpr()

        fh.write("TPR:{}\n".format(tpr))
        fh.write("FPR:{}\n".format(fpr))

    pdb.set_trace()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--attack", type=str, dest="atk_period")
    parser.add_argument("--conf", type=str, dest="conf")
    parser.add_argument("--malicious", type=str, dest="malicious")
    parser.add_argument("--benign", type=str, dest="infile")
    parser.add_argument("--cache", action="store_true", dest="cache")

    args = parser.parse_args()
    main(args.atk_period, args.conf, args.malicious, args.infile, args.cache)
