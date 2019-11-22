import argparse
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pdb
import yaml
import predicate as pred
import pvStore
from itemset import get_transactions
from idsInvariants import IDSInvariant
from timeChecker import TimeChecker
import utils

#MAL_CACHE = "malicious_activities.bin"
#MATRIX = "matrices.bin"
#LOG = "res.log"

MAL_CACHE = "malicious_cache"
MATRIX = "matrix"
TIME_LOG = "time_log"
OUTPUT_RES = "output_res"

TEST_INVARIANTS = "test_with_invariant"
INV_FILE = "invariants_file"
OUTPUT_RES_INV = "invariant_output"
INV_LOG = "invariant_log"

START = 'start'
END = 'end'
TS = 'timestamp'

class EvalResult(object):

    def __init__(self, tp, fp, tn, fn, nbr_attack, missed_atk=None,
                 false_atk=None):

        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.nbr_attack = nbr_attack
        self.missed_atk = missed_atk
        self.false_atk = false_atk

    def fpr(self):
        # Among everything that was detected, what is the ratio of false alert
        try:
            return self.fp/(self.tp + self.fp)
        except ZeroDivisionError:
            return -1

    def tpr(self):
        # Among all the existing attack, how much were you able to detect
        try:
            return self.tp/(self.tp + self.fn)
        except ZeroDivisionError:
            return -1

def invariant_comparison(mal_expected, mal_computed, attack_store, just_after):

    i = 0
    j = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    nbr_attack = 0

    attack_missed = set()
    attack_false = set()

    detect_timestamps = dict_list_ts(mal_computed, False)


    for state in attack_store:
        if i < len(mal_expected):
            attack = mal_expected[i]
        if j < len(detect_timestamps):
            detect = detect_timestamps[j]

        tmp = state[TS]
        ts = datetime(year=tmp.year, month=tmp.month, day=tmp.day,
                      hour=tmp.hour, minute=tmp.minute, second=tmp.second)
        key = (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)

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
            if not just_after:
                nbr_attack += 1

                if attack_detected:
                    true_positive += 1
                else:
                    false_negative += 1
                    attack_missed.add(key)
            else:
                if attack_detected:
                    false_positive += 1
                    attack_false.add(key)
                else:
                    true_negative += 1
            i += 1

        # When last attack was done
        elif ts > attack[END]:
            if attack_detected:
                false_positive += 1
                attack_false.add(key)
            else:
                true_negative += 1

    return EvalResult(true_positive, false_positive, true_negative,
                      false_negative, nbr_attack, attack_missed,
                      attack_false)

def get_pv_state(values, old_values, pv_store):
    pv_state = {} 
    for k, v in values.items():
        if k != "timestamp":
            if v in pv_store.vars[k].limit_values:
                pv_state[k] = v
            else:
                pv_state[k] = old_values[k]
    return pv_state

def count_number_transitions(old_values, new_values):
    nbr_transitions = 0
    for k, v in old_values.items():
        if v != new_values[k]:
            nbr_transitions += 1
    return nbr_transitions

def dict_list_ts(d, timePattern=True):
    try:
        if timePattern:
            return sorted([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in d.keys()])
        else:
            return sorted([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in d])
    except AttributeError:
        return sorted([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in d])

            
def time_pattern_comparison(mal_expected, mal_computed, attack_store, pv_store):

    """
        mal_expected: ts->expected number of malicious transition in timestamp
        mal_computed: ts->computed number of malicious transition in timestamp ts
    """
    i = 0
    j = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    expected_timestamps = dict_list_ts(mal_expected)
    detect_timestamps = dict_list_ts(mal_computed)

    last_state = None

    for state in attack_store:
        if i < len(expected_timestamps):
            attack = expected_timestamps[i]
        if j < len(detect_timestamps):
            detect = detect_timestamps[j]
        tmp = state[TS]
        ts = datetime(year=tmp.year, month=tmp.month, day=tmp.day,
                      hour=tmp.hour, minute=tmp.minute, second=tmp.second)
        key = (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)

        attack_detected = ts == detect
        if attack_detected:
            j += 1
            nbr_detect = len(mal_computed[ts])

        nbr_transitions = len(pv_store.vars)
        if last_state is not None:
            new_state = get_pv_state(state, last_state, pv_store)
            nbr_transitions = count_number_transitions(last_state, new_state)
        else:
            new_state = get_pv_state(state, state, pv_store)

        last_state = new_state

        if attack != ts:
            if attack_detected:
                false_positive += nbr_detect
                true_negative += max(0, nbr_transitions - nbr_detect)
            else:
                true_negative += nbr_transitions

        elif attack == ts:
            i += 1
            nbr_expect = mal_expected[ts]
            if attack_detected:
                true_positive += min(nbr_expect, nbr_detect)
                false_negative += max(0, nbr_expect - nbr_detect)
                #detect too much
                false_positive += max(0, nbr_detect - nbr_expect)
                true_negative += nbr_transitions - max(nbr_expect, nbr_detect)
            else:
                false_negative += nbr_expect
                true_negative += max(0, nbr_transitions - nbr_expect)

    return EvalResult(true_positive, false_positive, true_negative,
                      false_negative, 0)

def compare_activities(mal_expected, mal_computed, attack_store, pv_store,
                       timePattern=True, just_after=False):
    # Cannot comparte the timestamp sets because the expected time
    # are only range

    # just_after: the end time may represent the last time the attack occured.
    #           or the time just after the last occurence of the attack occured.
            
    if timePattern:
        res = time_pattern_comparison(mal_expected, mal_computed, attack_store, pv_store)
    else:
        res = invariant_comparison(mal_expected, mal_computed, attack_store, just_after)
    return res

def create_expected_malicious_activities(atk_period, timePattern=False):
    with open(atk_period) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        if not timePattern:
            for attack in desc:
                starttime = datetime.strptime(attack[START], '%d/%m/%Y %H:%M:%S')
                endtime = datetime.strptime(attack[END], '%d/%m/%Y %H:%M:%S')
                attack[START] = starttime
                attack[END] = endtime
        return desc

def run_invariant_ids(params, conf, data, data_mal, infile, malicious):
    if data is None:
        normal = utils.read_state_file(infile)
    else:
        normal = data

    if data_mal is None:
        malicious = utils.read_state_file(malicious)
    else:
        malicious = data_mal
    predicates = pred.generate_all_predicates(conf, normal)
    mapping_id_pred = {}
    _ = get_transactions(normal, predicates, mapping_id_pred)
    ids = IDSInvariant(mapping_id_pred, params[INV_FILE], params[INV_LOG])
    for state in malicious:
        ids.valid_state(state)
    ids.close()
    return ids

def main(atk_period_time, atk_period_inv, conf, malicious,
         infile, params, cache):

    data_mal = utils.read_state_file(malicious)
    data = None
    pv_store = pvStore.PVStore(conf)
    if not cache:
        data = utils.read_state_file(infile)
        time_checker = TimeChecker(conf, params[TIME_LOG], data)
        time_checker.fill_matrices()
        pickle.dump(time_checker.matrices, open(MATRIX, "wb"))
        time_checker.detection_store = data_mal
        time_checker.detect_suspect_transition()
        pickle.dump(time_checker.malicious_activities, open(params[MAL_CACHE], "wb"))
        time_checker.close()
        malicious_activities = time_checker.malicious_activities
        time_checker.create_matrices()
    else:
        matrices = pickle.load(open(params[MATRIX], "rb"))
        malicious_activities = pickle.load(open(params[MAL_CACHE], "rb"))

    expected_atk = create_expected_malicious_activities(atk_period_time, True)
    res = compare_activities(expected_atk, malicious_activities, data_mal,
                             pv_store, True, True)

    if params[TEST_INVARIANTS]:
        
        expected_atk = create_expected_malicious_activities(atk_period_inv)
        ids = run_invariant_ids(params, conf, data, data_mal, infile, malicious)

        inv_res = compare_activities(expected_atk, ids.malicious_activities,
                                     data_mal, pv_store, False, True)

    with open(params[OUTPUT_RES], "w") as fh:
        fh.write("Time based\n")
        fh.write("Total:{}\n".format(len(data_mal)))
        fh.write("Nbr Attack: {}\n".format(res.nbr_attack))
        fh.write("TP:{}\n".format(res.tp))
        fh.write("FP:{}\n".format(res.fp))
        fh.write("TN:{}\n".format(res.tn))
        fh.write("FN:{}\n".format(res.fn))

        fh.write("TPR:{}\n".format(res.tpr()))
        fh.write("FPR:{}\n".format(res.fpr()))

        if params[TEST_INVARIANTS]:
            fh.write("Invariant Based\n")
            fh.write("Total:{}\n".format(len(data_mal)))
            fh.write("Nbr Attack: {}\n".format(inv_res.nbr_attack))
            fh.write("TP:{}\n".format(inv_res.tp))
            fh.write("FP:{}\n".format(inv_res.fp))
            fh.write("TN:{}\n".format(inv_res.tn))
            fh.write("FN:{}\n".format(inv_res.fn))
            fh.write("TPR:{}\n".format(inv_res.tpr()))
            fh.write("FPR:{}\n".format(inv_res.fpr()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--atk-inv", type=str, dest="atk_period_inv",
                        help="File containing the period time of the attacks for invariant IDS")
    parser.add_argument("--atk-time", type=str, dest="atk_period_time",
                        help="File containing the time of the attacks for time IDS")
    parser.add_argument("--conf", type=str, dest="conf",
                        help="File with the description of the process variable")
    parser.add_argument("--malicious", type=str, dest="malicious",
                        help="Timeseries of the scenario where an attack occured")
    parser.add_argument("--benign", type=str, dest="infile",
                        help="Timeseries of the scenario where no attack occured")
    parser.add_argument("--cache", action="store_true", dest="cache",
                        help="Should we use data previously computed")
    parser.add_argument("--params", type=str, dest="params",
                        help="Configuration file for the evaluation")

    args = parser.parse_args()
    with open(args.params, "r") as fname:
        params = yaml.load(fname, Loader=yaml.BaseLoader)

    main(args.atk_period_time, args.atk_period_inv, args.conf, args.malicious,
         args.infile, params, args.cache)
