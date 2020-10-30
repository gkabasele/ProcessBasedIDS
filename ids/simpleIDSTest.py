import pdb
import argparse
import pickle
import os
from datetime import datetime
from datetime import timedelta

import utils
import pvStore
import evaluationIDS
from idsAR import IDSAR
from timeChecker import TimeChecker
import predicate as pred
from itemset import get_transactions
from idsInvariants import IDSInvariant
import numpy as np

TIMEPAT = "timepattern"
AR = "ar"
INV = "invariant"

def setup(inputfile, attackfile, conf):

    print("Loading all the states from the systems")

    data = utils.read_state_file(inputfile)
    pv_store = pvStore.PVStore(conf, data)
    data_atk = utils.read_state_file(attackfile)

    return pv_store, data, data_atk

def run_ar_ids(pv_store, data, data_atk):
    ids = IDSAR(pv_store, data, control_coef=6, alpha=0.02)
    ids.create_predictors()
    ids.train_predictors()

    filename = "./ar_malicious.bin"
    file_reason = "./ar_reasons.bin"
    if os.path.exists(filename) and os.path.exists(file_reason):
        with open(filename, "rb") as f:
            ids.malicious_activities = pickle.load(f)

        with open(file_reason, "rb") as f:
            ids.malicious_reason = pickle.load(f)
    else:
        ids.run_detection_mode(data_atk)
        with open(filename, "wb") as f:
            pickle.dump(ids.malicious_activities, f)

        with open(file_reason, "wb") as f:
            pickle.dump(ids.malicious_reason, f)

    ids.export_detected_atk("useless_logfile_ar.txt")

    return ids

def run_time_pattern_ids(pv_store, data, data_atk, matrix, write_matrix):
    export_log = "useless_logfile_pattern.txt"
    print("Configuring matrix")
    ids = TimeChecker(data, pv_store,
                      export_log, noisy=True)
    export = False
    if not write_matrix:
        ids.import_matrix(matrix)
    else:
        ids.create_matrices()
        ids.fill_matrices()
        ids.export_matrix(matrix)


    print("Running IDS on attack trace")
    ids.detection_store = data_atk

    filename = "./tp_malicious.bin"

    if os.path.exists(filename):
        export = True
        with open(filename, "rb") as f:
            ids.malicious_activities = pickle.load(f)
    else:
        ids.detect_suspect_transition()
        with open(filename, "wb") as f:
            pickle.dump(ids.malicious_activities, f)

    ids.close()

    if export:
        ids.export_detected_atk(export_log)

    return ids

def run_invariant_ids(pv_store, data_atk, pred_file,
                      map_id_pred_file, inv_file):

    export = False
    if pred_file is not None and map_id_pred_file is not None:
        with open(pred_file, "rb") as fname:
            predicates = pickle.load(fname)

        with open(map_id_pred_file, "rb") as fname:
            map_id_pred = pickle.load(fname)

    else:
        raise ValueError("Missing file to run the invariant IDS")

    export_log = "useless_logifle_invariant.txt"

    ids = IDSInvariant(map_id_pred, inv_file, export_log)

    filename = "./inv_malicious.bin"

    if os.path.exists(filename):
        export = True
        with open(filename, "rb") as f:
            ids.malicious_activities = pickle.load(f)
    else:
        ids.run_detection(data_atk, pv_store.continuous_monitor_vars(), predicates, map_id_pred)
        with open(filename, "wb") as f:
            pickle.dump(ids.malicious_activities, f)

    ids.close()
    if export:
        ids.export_detected_atk(export_log)

    return ids


def get_ids_result(ids, time_atk, windows=0):
    # window to wait to consider that an alert is
    #linked to an attack

    relaxed_win = timedelta(seconds=windows)

    i = 0

    alerts = list(ids.malicious_activities.keys())
    alerts_var = list(ids.malicious_activities.values())
    #inter false positive time
    fp_start = None
    ifpt = list()

    detect_in_period = dict()

    wrong_alert = 0

    for idx, alert in enumerate(alerts):
        #-----[*********]---[**]--- (period)
        #-------*****--------*--  (alert)
        # find next period
        if alert > time_atk[i]["end"]:
            while i < len(time_atk) - 1 and alert > time_atk[i]["end"]:
                i += 1

            if i == len(time_atk) - 1 and alert > time_atk[i]["end"] + relaxed_win:
                #pdb.set_trace()
                wrong_alert += 1
                if fp_start is not None:
                    ifpt.append((alert - fp_start).total_seconds())
                fp_start = alert

        #-----[*********]------ (period)
        #-**----*****---------  (alert)
        if alert < time_atk[i]["start"]:
            if i == 0 or (i > 0 and alert > time_atk[i-1]["end"] + relaxed_win):
                #pdb.set_trace() 
                wrong_alert += 1
                if fp_start is not None:
                    ifpt.append((alert - fp_start).total_seconds())
                fp_start = alert

        start_p = time_atk[i]["start"]
        end_p = time_atk[i]["end"]

        if alert >= start_p and alert <= end_p:
            if time_atk[i]["start"] not in detect_in_period:
                detect_in_period[time_atk[i]["start"]] = (alert - start_p).total_seconds()


    miss_atk = len(time_atk) - len(detect_in_period)
    return detect_in_period, wrong_alert, miss_atk, ifpt

def run_ids_eval(func_ids, name, atk_file, window, *args):

    ids = func_ids(*args)
    detected, wrong_alert, miss_atk, inter_fp_time = get_ids_result(ids, atk_file, window)

    detection_time = list(detected.values())
    mean_det = np.mean(detection_time)
    std_det = np.std(detection_time)
    minval_det = np.min(detection_time)
    maxval_det = np.max(detection_time)

    mean_ifpt = np.mean(inter_fp_time)
    std_ifpt = np.std(inter_fp_time)
    minval_ifpt = np.min(inter_fp_time)
    maxval_ifpt = np.max(inter_fp_time)

    print("{}".format(name))
    print("------------")
    print("Detected: {}".format(len(detected)))
    print("Wrong alert: {}".format(wrong_alert))
    print("Miss Attack: {}".format(miss_atk))
    print("Detection Time Mean/Std/Min/Max: {}/{}/{}/{}".format(mean_det, std_det,
                                                                minval_det, maxval_det))

    print("Inter FP Time Mean/Std/Min/Max: {}/{}/{}/{}".format(mean_ifpt, std_ifpt,
                                                               minval_ifpt, maxval_ifpt))
    print("------------")

def main(inputfile, attackfile, conf, atk_time_file, run_ids,
         matrix, write, pred_file, map_id_pred, inv_file):

    pv_store, data, data_atk = setup(inputfile, attackfile, conf)
    atk_file = evaluationIDS.create_expected_malicious_activities(atk_time_file)

    if TIMEPAT in run_ids:
        run_ids_eval(run_time_pattern_ids, TIMEPAT, atk_file, 30, pv_store, data,
                     data_atk, matrix, write)

    if AR in run_ids:
        run_ids_eval(run_ar_ids, AR, atk_file, 0, pv_store, data, data_atk)

    if INV in run_ids:
        run_ids_eval(run_invariant_ids, INV, atk_file, 0, pv_store, data_atk, pred_file,
                     map_id_pred, inv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf")
    parser.add_argument("--input", action="store", dest="input_file")
    parser.add_argument("--attack", action="store", dest="attack_file")
    parser.add_argument("--matrix", action="store", dest="matrix")
    parser.add_argument("--time", action="store", dest="atk_time")
    parser.add_argument("--runids", action="store", nargs='+', type=str,
                        dest="runids", default=TIMEPAT)
    parser.add_argument("--write", action="store_true", default=False, dest="write")

    parser.add_argument("--predicates", action="store", dest="pred_file")
    parser.add_argument("--map", action="store", dest="map_id_pred")
    parser.add_argument("--invariants", action="store", dest="inv_file")

    args = parser.parse_args()

    main(args.input_file, args.attack_file, args.conf, args.atk_time,
         args.runids, args.matrix, args.write, args.pred_file,
         args.map_id_pred, args.inv_file)
