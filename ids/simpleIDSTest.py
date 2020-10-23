import pdb
import argparse

import utils
import pvStore
import evaluationIDS
from idsAR import IDSAR
from timeChecker import TimeChecker

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
    ids = IDSAR(pv_store, data, alpha=0.02)
    ids.create_predictors()
    ids.train_predictors()

    pdb.set_trace()

    ids.run_detection_mode(data_atk)
    ids.export_detected_atk(data_atk)

    return ids

def run_time_pattern_ids(pv_store, data, data_atk, matrix, write_matrix):

    print("Configuring matrix")
    ids = TimeChecker(data, pv_store,
                      "useless_logifle.txt", noisy=True)

    if not write_matrix:
        ids.import_matrix(matrix)
    else:
        ids.create_matrices()
        ids.fill_matrices()
        ids.export_matrix(matrix)

    pdb.set_trace()

    print("Running IDS on attack trace")
    ids.detection_store = data_atk
    ids.detect_suspect_transition()

    ids.close()

    return ids

def get_ids_result(alerts, time_atk):

    i = 0

    detect_in_period = set()

    wrong_alert = 0

    for alert in alerts:
        #-----[*********]---[**]--- (period)
        #-------*****--------*--  (alert)
        # find next period
        if alert > time_atk[i]["end"]:
            while i < len(time_atk) - 1 and alert > time_atk[i]["end"]:
                i += 1

            if i == len(time_atk) - 1 and alert > time_atk[i]["end"]:
                wrong_alert += 1


        #-----[*********]------ (period)
        #-**----*****---------  (alert)
        if alert < time_atk[i]["start"]:
            wrong_alert += 1

        start_p = time_atk[i]["start"]
        end_p = time_atk[i]["end"]

        if alert >= start_p and alert <= end_p:
            detect_in_period.add(time_atk[i]["start"])


    miss_atk = len(time_atk) - len(detect_in_period)
    return len(detect_in_period), wrong_alert, miss_atk



def main(inputfile, attackfile, conf, atk_time_file, run_ids, matrix, write):

    pv_store, data, data_atk = setup(inputfile, attackfile, conf)
    atk_file = evaluationIDS.create_expected_malicious_activities(atk_time_file)

    if TIMEPAT in run_ids:
        ids_tp = run_time_pattern_ids(pv_store, data, data_atk, matrix, write)
        res = get_ids_result(ids_tp.malicious_activities.keys(), atk_file)

    if AR in run_ids:
        ids_ar = run_ar_ids(pv_store, data, data_atk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf")
    parser.add_argument("--input", action="store", dest="inputfile")
    parser.add_argument("--attack", action="store", dest="attackfile")
    parser.add_argument("--matrix", action="store", dest="matrix")
    parser.add_argument("--time", action="store", dest="atk_time")
    parser.add_argument("--runids", action="store", nargs='+', type=str,
                        dest="runids", default=TIMEPAT)
    parser.add_argument("--write", action="store_true", default=False, dest="write")

    args = parser.parse_args()

    main(args.inputfile, args.attackfile, args.conf, args.atk_time,
         args.runids, args.matrix, args.write)
