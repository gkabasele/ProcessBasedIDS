import pdb
import argparse

import utils
import pvStore
import evaluationIDS
from idsAR import IDSAR
from timeChecker import TimeChecker


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

def main(inputfile, attackfile, conf, atk_time_file, matrix, write):
    atk_file = evaluationIDS.create_expected_malicious_activities(atk_time_file)

    pdb.set_trace()

    pv_store, data, data_atk = setup(inputfile, attackfile, conf)

    ids_tp = run_time_pattern_ids(pv_store, data, data_atk, matrix, write)
    ids_ar = run_ar_ids(pv_store, data, data_atk)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf")
    parser.add_argument("--input", action="store", dest="inputfile")
    parser.add_argument("--attack", action="store", dest="attackfile")
    parser.add_argument("--matrix", action="store", dest="matrix")
    parser.add_argument("--time", action="store", dest="atk_time")
    parser.add_argument("--write", action="store_true", default=False, dest="write")

    args = parser.parse_args()

    main(args.inputfile, args.attackfile, args.conf, args.atk_time,
         args.matrix, args.write)
