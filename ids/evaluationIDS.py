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

TEST_TIMEPATTERN = "test_with_time_pattern"
MAL_CACHE = "malicious_cache"
MATRIX = "matrix"
TIME_LOG = "time_log"
OUTPUT_RES = "output_res"
OUTPUT_ANALYSIS = "output_analysis"
ATK_TIME_TIMEPAT = "attack_time_time_pattern"

TEST_INVARIANTS = "test_with_invariant"
INV_FILE = "invariants_file"
OUTPUT_RES_INV = "invariant_output"
INV_LOG = "invariant_log"
PRED_MAP = "predicate_map_file"
PREDICATES = "predicates"
GEN_PRED = "generate_predicate"
ATK_TIME_INV = "attack_time_inv"

VAR_STORE = "variable_store"

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
            if self.fp == 0:
               return 0 
            return -1

    def tpr(self):
        # Among all the existing attack, how much were you able to detect
        try:
            return self.tp/(self.tp + self.fn)
        except ZeroDivisionError:
            if self.tp == 0:
                return 0
            return -1

    def __str__(self):
        return "TPR:{}, FPR:{}".format(self.tpr(), self.fpr())

    def __repr__(self):
        return self.__str__()

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

    attack = None
    detect = None

    if len(mal_expected) == 0:
        for state in attack_store:
            if j < len(mal_computed):
                detect = detect_timestamps[j]
                tmp = state[TS]
                ts = datetime(year=tmp.year, month=tmp.month, day=tmp.day,
                              hour=tmp.hour, minute=tmp.minute, second=tmp.second)
                key = (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)

                attack_detected = ts == detect

                if attack_detected:
                    j += 1
                    false_positive += 1
    else:
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
        if k != "timestamp" and k != "normal/attack":
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

def time_pattern_comp_v2(mal_expected, mal_computed, attack_store, pv_store):
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

    nbr_attack = 0
    attack_missed = set()
    attack_false = set()

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

        if attack == ts:
            i += 1
            if type(mal_expected[ts]) == int:
                nbr_expect = mal_expected[ts]
            else:
                atk_info = mal_expected[ts].split(",")
                nbr_expect = int(atk_info[0])

            # No attack in progress
            if nbr_expect == 0:
                if attack_detected:
                    attack_false.add(key)
                    false_positive += 1
                else:
                    true_negative += 1
            else:
                nbr_attack += 1
                if attack_detected:
                    true_positive += 1
                else:
                    attack_missed.add(key)
                    false_negative += 1

    return EvalResult(true_positive, false_positive, true_negative,
            false_negative, nbr_attack, attack_missed, attack_false)

            
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

    nbr_attack = 0
    attack_missed = set()
    attack_false = set()

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
        if attack == ts:
            i += 1
            nbr_expect = mal_expected[ts]
            
            # No attack in progress
            if nbr_expect == 0:
                if attack_detected:
                    attack_false.add(key)
                    false_positive += nbr_detect
                    true_negative += max(0, nbr_transitions - nbr_detect)
                else:
                    true_negative += nbr_transitions
            else:
                if nbr_expect > 0:
                    nbr_attack += 1
                else:
                    raise ValueError("Unexpected negative value for the number malicious transition")


                if attack_detected:
                    true_positive += min(nbr_expect, nbr_detect)
                    false_negative += max(0, nbr_expect - nbr_detect)
                    #detect too much
                    false_positive += max(0, nbr_detect - nbr_expect)
                    true_negative += nbr_transitions - max(nbr_expect, nbr_detect)
                else:
                    false_negative += nbr_expect
                    true_negative += max(0, nbr_transitions - nbr_expect)
                    attack_missed.add(key)

    return EvalResult(true_positive, false_positive, true_negative,
                      false_negative, nbr_attack, attack_missed, 
                      attack_false)

def compare_activities(mal_expected, mal_computed, attack_store, pv_store,
                       timePattern=True, just_after=False):
    # Cannot comparte the timestamp sets because the expected time
    # are only range

    # just_after: the end time may represent the last time the attack occured.
    #           or the time just after the last occurence of the attack occured.

    if timePattern:
        #res = time_pattern_comparison(mal_expected, mal_computed, attack_store, pv_store)
        res = time_pattern_comp_v2(mal_expected, mal_computed, attack_store, pv_store)
    else:
        res = invariant_comparison(mal_expected, mal_computed, attack_store, just_after)
    return res

def create_expected_malicious_activities(atk_period, timePattern=False):
    with open(atk_period) as fh:
        content = fh.read()
        desc = yaml.load(content, Loader=yaml.Loader)
        # There is not atk in the trace
        if desc is None:
            return []

        if not timePattern:
            for attack in desc:
                starttime = datetime.strptime(attack[START], '%d/%m/%Y %H:%M:%S')
                endtime = datetime.strptime(attack[END], '%d/%m/%Y %H:%M:%S')
                attack[START] = starttime
                attack[END] = endtime
        return desc

def run_invariant_ids(params, conf, store, data, data_mal, infile, malicious):
    if data is None:
        normal = utils.read_state_file(infile)
    else:
        normal = data

    if data_mal is None:
        malicious = utils.read_state_file(malicious)
    else:
        malicious = data_mal

    sensors = store.continuous_monitor_vars()

    if params[GEN_PRED]:
        predicates = pred.generate_all_predicates(conf, normal)
        mapping_id_pred = {}
        _ = get_transactions(normal, sensors, predicates, mapping_id_pred, False)

    else:
        with open(params[PREDICATES], "rb") as fname:
            predicates = pickle.load(fname)

        with open(params[PRED_MAP], "rb") as fname:
            mapping_id_pred = pickle.load(fname)


    ids = IDSInvariant(mapping_id_pred, params[INV_FILE], params[INV_LOG])
    for i, state in enumerate(malicious):
        if i % 50000 == 0:
            print("Up to state: {}".format(i))
        ids.valid_state(state, sensors, predicates, mapping_id_pred)
    ids.close()
    return ids

def export_ids_result(filename, analysis_filename, detection_method,
                      data_mal, res):
    with open(filename, "w") as fh:
        fh.write("{}\n".format(detection_method))
        fh.write("Total:{}\n".format(len(data_mal)))
        fh.write("Nbr Attack: {}\n".format(res.nbr_attack))
        fh.write("TP:{}\n".format(res.tp))
        fh.write("FP:{}\n".format(res.fp))
        fh.write("TN:{}\n".format(res.tn))
        fh.write("FN:{}\n".format(res.fn))
        fh.write("TPR:{}\n".format(res.tpr()))
        fh.write("FPR:{}\n".format(res.fpr()))
        fh.write("\t----")

    with open(analysis_filename, "w") as fh:
        fh.write("False positive timestamp\n")
        fh.write("--------------------------\n")
        attack_false = [x for x in res.false_atk]
        attack_false.sort()
        for ts in attack_false:
            fh.write(str(ts) + "\n")

        fh.write("\n\n")
        fh.write("Missed attack timestamp\n")
        fh.write("--------------------------\n")
        attack_missed = [x for x in res.missed_atk]
        attack_missed.sort()
        for ts in attack_missed:
            fh.write(str(ts) + "\n")

def export_state_from_ts(filename, timestamps, data):
    with open(filename + ".txt", "w") as fname:
        i = 0
        states = []
        for state in data:
            ts = state[TS]
            fp_time = datetime.strptime(timestamps[i], "%d-%m-%Y %H:%M:%S")
            if ts == fp_time:
                states.append(state)
                fname.write(str(state) + "\n")
                i += 1
            if i >= len(timestamps):
                break

    with open(filename + ".bin", "wb") as fname:
        pickle.dump(states, fname)

def test_invariants(params, store, state_filename):

    with open(params[PREDICATES], "rb") as fname:
        predicates = pickle.load(fname)
    with open(params[PRED_MAP], "rb") as fname:
        mapping_id_pred = pickle.load(fname)

    sensors = store.continuous_monitor_vars()

    with open("./valid_state_for_texting.bin", "rb") as fname:
        state = pickle.load(fname)
        state["lit301"] = 1500.0
    ids = IDSInvariant(mapping_id_pred, params[INV_FILE], "one_state.log")
    ids.valid_state(state, sensors, predicates, mapping_id_pred)
    ids.close()
    print(len(ids.malicious_activities) == 1)

def main(atk_period_time, atk_period_inv, conf, malicious,
         infile, params, cache):

    print("Loading all the states from the systems trace")
    data_mal = utils.read_state_file(malicious)
    data = utils.read_state_file(infile)

    #FIXME remove cache args.
    print("Importing process variables")
    if not cache:
        if not os.path.exists(params[VAR_STORE]):
            pv_store = pvStore.PVStore(conf, data)
            with open(params[VAR_STORE], "wb") as fname:
                pickle.dump(pv_store, fname)
        else:
            with open(params[VAR_STORE], "rb") as fname:
                pv_store = pickle.load(fname)

    else:
        with open(params[VAR_STORE], "rb") as fname:
            pv_store = pickle.load(fname)

    pdb.set_trace()

    print("Creating time attacks")
    if not os.path.exists(params[ATK_TIME_TIMEPAT]):
        expected_atk = create_expected_malicious_activities(atk_period_time, True)
        with open(params[ATK_TIME_TIMEPAT], "wb") as fname:
            pickle.dump(expected_atk, fname)
    else:
        with open(params[ATK_TIME_TIMEPAT], "rb") as fname:
            expected_atk = pickle.load(fname)

    if params[TEST_TIMEPATTERN]:
        print("Running time pattern IDS")
        if not cache:
            print("Creating matrices")
            time_checker = TimeChecker(conf, params[TIME_LOG], data)
            if not os.path.exists(params[MATRIX]):
                time_checker.create_matrices()
                time_checker.fill_matrices()
                with open(params[MATRIX], "wb") as fname:
                    pickle.dump(time_checker.matrices, fname)
            else:
                with open(params[MATRIX], "rb") as fname:
                    time_checker.matrices = pickle.load(fname)
            time_checker.detection_store = data_mal
            pdb.set_trace()
            print("Running detection")
            if not os.path.exists(params[MAL_CACHE]):
                time_checker.detect_suspect_transition()
                with open(params[MAL_CACHE], "wb") as fname:
                    pickle.dump(time_checker.malicious_activities, fname)
                time_checker.close()
                malicious_activities = time_checker.malicious_activities
            else:
                with open(params[MAL_CACHE], "rb") as fname:
                    malicious_activities = pickle.load(fname)

        else:
            with open(params[MATRIX], "rb") as fname:
                time_checker = pickle.load(fname)

            with open(params[MAL_CACHE], "rb") as fname:
                malicious_activities = pickle.load(fname)

        pdb.set_trace()

        res = compare_activities(expected_atk, malicious_activities, data_mal,
                                 pv_store, True, True)

        export_ids_result(params[OUTPUT_RES], params[OUTPUT_ANALYSIS], "Time based", data_mal, res)
        pdb.set_trace()

    if params[TEST_INVARIANTS]:
        print("Running invariant IDS")
        #dates_false = ["28-12-2015 11:33:32", "28-12-2015 11:36:32", "28-12-2015 12:00:34"]
        #export_state_from_ts("./false_positive_state", dates_false, data_mal)

        expected_atk = create_expected_malicious_activities(atk_period_inv)

        ids = run_invariant_ids(params, conf, pv_store, data, data_mal, infile, malicious)

        inv_res = compare_activities(expected_atk, ids.malicious_activities,
                                     data_mal, pv_store, False, True)
        print("Exporting evaluation result")
        export_ids_result(params[OUTPUT_RES], params[OUTPUT_ANALYSIS],
                          "Invariant", data_mal, inv_res)


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
        for k, v in params.items():
            if v == 'False' or v == "True":
                params[k] = v == "True"

    main(args.atk_period_time, args.atk_period_inv, args.conf, args.malicious,
         args.infile, params, args.cache)
