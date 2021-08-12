import pdb
import argparse
import pickle
import os
from collections import Counter, OrderedDict
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

ATKTYPE = [
            "actuatorMod",
            "sensorRewritting",
            "sensorBlock",
            "outOfBounds",
            "actuatorBlock"
          ]

class IDSResultData(object):

    def __init__(self, detected=None, wrong_alert=None, miss_atk=None, inter_fp_time=None,
                 eval_res=None, atk_types=None, detect_types=None, false_alert_repartition=None,
                 true_alert_repartition=None):

        self.detected = detected
        self.wrong_alert = wrong_alert
        self.miss_atk = miss_atk
        self.inter_fp_time = inter_fp_time
        self.eval_res = eval_res
        self.atk_types = atk_types
        self.detect_types = detect_types
        self.false_alert_repartition = false_alert_repartition
        self.true_alert_repartition = true_alert_repartition

def setup(inputfile, attackfile, conf, pvstore_file, create_pv):

    print("Loading all the states from the systems")

    data = utils.read_state_file(inputfile)
    if create_pv:
        pv_store = pvStore.PVStore(conf, data)
        with open(pvstore_file, "wb") as f:
            pickle.dump(pv_store, f)
    else:
        with open(pvstore_file, "rb") as f:
            pv_store = pickle.load(f)
    data_atk = utils.read_state_file(attackfile)

    return pv_store, data, data_atk

def run_ar_ids(pv_store, data, data_atk):
    #ids = IDSAR(pv_store, data, control_coef=3, alpha=0.1, maxorder=512)
    ids = IDSAR(pv_store, data, control_coef=3, alpha=0.1, maxorder=120)


    filename = "./ar_malicious_swat.bin"
    file_reason = "./ar_reasons_swat.bin"
    
    #filename = "./ar_malicious_swat0801.bin"
    #file_predictor = "./ar_predictor_swat_fscore.bin"
    #file_reason = "./ar_reasons_swat.bin"

    #filename = "./ar_malicious_medium_fscore_equalvar.bin"
    #file_reason = "./ar_reasons_medium_fscore_equalvar.bin"
    #file_predictor = "./ar_predictor_medium_fscore_equalvar.bin"

    #file_predictor = "./ar_predictor_medium_process_maxorder120_overflow.bin"
    #filename = "./ar_malicious_medium_process_maxorder120_overflow.bin"
    #file_reason = "./ar_reasons_medium_process_maxorder120_overflow.bin"

    #file_predictor = "./ar_predictore_medium_process_maxorder120.bin"
    #filename = "./ar_malicious_medium_process_maxorder120.bin"
    #file_reason = "./ar_reasons_medium_process_maxorder120.bin"

    #file_predictor = "./ar_predictore_medium_process_maxorder120_fscore.bin"
    #filename = "./ar_malicious_medium_process_maxorder120_fscore.bin"
    #file_reason = "./ar_reasons_medium_process_maxorder120_fscore.bin"

    #file_predictor = "./ar_predictore_medium_process_maxorder512_fscore.bin"   
    #filename = "./ar_malicious_medium_process_maxorder512_fscore.bin"          
    #file_reason = "./ar_reasons_medium_process_maxorder512_fscore.bin"

    if os.path.exists(filename) and os.path.exists(file_reason):
        with open(filename, "rb") as f:
            print("Reading malicious activities time from file " + filename)
            ids.malicious_activities = pickle.load(f)

        with open(file_reason, "rb") as f:
            ids.malicious_reason = pickle.load(f)
    else:
        
        if not os.path.exists(file_predictor):
            print("Training autoregressive IDS")
            ids.create_predictors()
            ids.train_predictors()
            ids.export_model(file_predictor)
        else:
            print("Importing autoregressive predictor")
            ids.import_model(file_predictor)

        print("Running IDS on attack trace")
        ids.run_detection_mode(data_atk)
        with open(filename, "wb") as f:
            pickle.dump(ids.malicious_activities, f)

        with open(file_reason, "wb") as f:
            pickle.dump(ids.malicious_reason, f)

    #ids.export_detected_atk("./useless_logfile_ar_medium_overflow.txt")
    #ids.export_detected_atk("./useless_logfile_ar_swat_fscore.txt")
    ids.export_detected_atk("./useless_logfile_ar_swat.txt")
    #ids.export_detected_atk("./useless_logfile_ar_swat_0801.txt")

    return ids

def run_time_pattern_ids(pv_store, data, data_atk, matrix,
                         write_matrix, strategy, tp_filename,
                         trans_file):
    print("Running Time pattern IDS")
    #export_log = "./useless_logfile_pattern_swat_barrier_basic_detection.txt"
    export_log = "./useless_logfile_pattern_swat_barrier0801.txt"
    export_log = "./useless_logfile_pattern_medium_barrier_backup_overflow.txt"
    print("Configuring matrix")
    ids = TimeChecker(data, pv_store,
                      export_log, strategy, noisy=True)
    export = False
    if not write_matrix:
        ids.import_matrix(matrix)
    else:
        ids.create_matrices()
        ids.fill_matrices()
        ids.export_matrix(matrix)

    ids.detection_store = data_atk
    
    filename = tp_filename
    #filename = "./tp_malicious_swat_0801_max_lof_duplication_multiplier_30.bin"

    if os.path.exists(filename):
        print("Reading malicious activities time from file " + filename)
        export = True
        with open(filename, "rb") as f:
            ids.malicious_activities = pickle.load(f)
    else:
        print("Running IDS on attack trace")
        ids.detect_suspect_transition()
        with open(filename, "wb") as f:
            pickle.dump(ids.malicious_activities, f)

    print("{}\n".format(ids.get_vars_alerts_hist()))

    ids.close()

    if export:
        ids.export_detected_atk(export_log)

    filename = trans_file
    if os.path.exists(filename):
        print("Reading transition log from file " + filename)
        ids.import_transition_log(filename)
    else:
        ids.export_transition_log(filename)


    return ids

def run_invariant_ids(pv_store, data_atk, pred_file,
                      map_id_pred_file, inv_file):

    print("Invariant based IDS")
    export = False
    if pred_file is not None and map_id_pred_file is not None:
        with open(pred_file, "rb") as fname:
            predicates = pickle.load(fname)

        with open(map_id_pred_file, "rb") as fname:
            map_id_pred = pickle.load(fname)

    else:
        raise ValueError("Missing file to run the invariant IDS")

    export_log = "./useless_logfile_invariant_swat0801.txt"
    #export_log = "./useless_logfile_invariant_medium_overflow.txt"

    ids = IDSInvariant(map_id_pred, inv_file, export_log)

    filename = "./inv_malicious_swat0801.bin"
    #filename = "./inv_malicious_medium_overflow.bin"

    if os.path.exists(filename):
        print("Reading malicious activities time from file " + filename)
        export = True
        with open(filename, "rb") as f:
            ids.malicious_activities = pickle.load(f)
    else:
        print("Running detection on attack trace")
        ids.run_detection(data_atk, pv_store.continuous_monitor_vars(), predicates, map_id_pred)
        with open(filename, "wb") as f:
            pickle.dump(ids.malicious_activities, f)

    ids.close()
    if export:
        ids.export_detected_atk(export_log)

    return ids


def get_nbr_attack(time_atk):
    nbr_attack = 0

    for period in time_atk:
        nbr_attack += int((period["end"] - period["start"]).total_seconds())

    return nbr_attack

def get_atk_type_nbr(time_atk, types):

    atk_types = {x: 0 for x in types}
    for atk in time_atk:
        for t in atk["attackType"]:
            atk_types[t] += 1

    return atk_types

def get_atk_type_detection_repartition(atk_types, atk_detected_type):

    repartion = dict()
    for x in atk_types:
        try:
            repartion[x] = atk_detected_type[x]/atk_types[x]
        except ZeroDivisionError:
            repartion[x] = None

    return repartion

def counter_distribution(data):
    c = Counter(data)
    total = sum(c.values(), 0.0)
    for k in c:
        c[k] /= total
    return c

def get_first_transition(time_atk, transition_log):
    #transition index
    i = 0
    j = 0

    # list of dictionnaries containing the first transitions
    # after an attack
    map_transition = list()


    while i < len(time_atk)-2:
        # get the legitimat period after an attack
        start_period = time_atk[i]["end"]
        end_period = time_atk[i+1]["start"]
        first_transitions = OrderedDict()

        if j < len(transition_log):

            transition = transition_log[j]
            while j < len(transition_log) and transition.end < end_period:

                transition = transition_log[j]
                # The transition must end after the attack
                if (transition.varname not in first_transitions and
                        transition.end > start_period):

                    first_transitions[transition.varname] = transition

                j += 1

            map_transition.append(first_transitions)

        i += 1

    #return [x for first_transitions in map_transition for x in first_transitions.values()]
    return map_transition


def has_impacted_transitions(transitions, start_period, end_period, cur_i):

    exceed = False


    for i, transition in enumerate(transitions[cur_i:]):
        trans_var = transition.values()

        for t_entry in transition.values():

            if end_period is not None:
                if t_entry.end > start_period and t_entry.end < end_period:
                    return cur_i + i + 1, True
                elif t_entry.end > end_period:
                    exceed = True
                    break
            else:
                if t_entry.end > start_period:
                    return cur_i + i, True
        if exceed:
            return cur_i + i, False
    return cur_i, False

def unique_impact_transitions(all_transitions, first_transitions):
    # {var : {(from,to): [point1, point2, ...]}}
    split_transitions = dict()
    unique_transitions = list()
    for t in all_transitions:
        t_key = (t.from_val, t.to_val)
        if t.varname in split_transitions:
            if t_key in split_transitions[t.varname]:
                split_transitions[t.varname][t_key].append(t.trans_time)

            else:
                split_transitions[t.varname][t_key] = [t.trans_time]
        else:
            split_transitions[t.varname] = {t_key : [t.trans_time]}

    for no_atk_period in first_transitions:
        new_period = OrderedDict()
        for t in no_atk_period.values():
            occurence = 0
            for point in split_transitions[t.varname][(t.from_val, t.to_val)]:
                if abs(t.trans_time - point) <= 60:
                    occurence += 1
            if occurence < 3:
                new_period[t.varname] = t
        unique_transitions.append(new_period)
    return unique_transitions

def get_ids_result(ids, time_atk, atk_trace_size, map_transition=None,
                   aftereffect=10, window=0,
                   is_invariant=False, is_time_pattern=False):

    # window to wait to consider that an alert is
    #linked to an attack
    relaxed_win = timedelta(seconds=window)

    transition_i = 0

    i = 0

    alerts = list(ids.malicious_activities.keys())
    alerts_var = list(ids.malicious_activities.values())

    false_alert_vars = list()
    true_alert_vars = list()

    #Period counting of attack
    #inter false positive time
    fp_start = None
    ifpt = list()

    detect_in_period = dict()

    detect_period_atk_type = dict()

    wrong_alert = 0

    #Second per second counting
    true_positive = 0
    false_positive = 0
    nbr_attack = get_nbr_attack(time_atk)

    atk_types = get_atk_type_nbr(time_atk, ATKTYPE)
    detect_type_repartition = {x:0 for x in ATKTYPE}

    for alert, alert_var in zip(alerts, alerts_var):
        #-----[*********]---[**]--- (period)
        #-------*****--------*--  (alert)
        # find next period
        if alert > time_atk[i]["end"]:
            while i < len(time_atk) - 1 and alert > time_atk[i]["end"]:
                i += 1
            # Some attacks does not have after effect
            if int((time_atk[i]["end"] - time_atk[i]["start"]).total_seconds()) > aftereffect:
                tmp_window = relaxed_win
            else:
                tmp_window = timedelta(seconds=0)

            if i == len(time_atk) - 1 and alert > time_atk[i]["end"] + tmp_window:
                impacted = False
                if is_time_pattern and map_transition is not None:
                    transition_i, impacted = has_impacted_transitions(map_transition,
                                                                      time_atk[i]["end"],
                                                                      None, transition_i)
                if not impacted:     
                    wrong_alert += 1
                    for v in alert_var:
                        false_alert_vars.append(v)
                    if fp_start is not None:
                        ifpt.append((alert - fp_start).total_seconds())
                    fp_start = alert

        #-----[*********]------ (period)
        #-**----*****---------  (alert)
        if alert < time_atk[i]["start"]:
            if int((time_atk[max(0, i-1)]["end"] - time_atk[max(0, i-1)]["start"]).total_seconds()) > aftereffect:
                tmp_window = relaxed_win
            else:
                tmp_window = timedelta(seconds=0)

            if i == 0 or (i > 0 and alert > time_atk[i-1]["end"] + tmp_window):
                impacted = False
                if is_time_pattern and map_transition is not None:
                    end_period = None
                    if i < len(time_atk) - 1:
                        end_period = time_atk[i+1]["start"]
                    transition_i, impacted = has_impacted_transitions(map_transition,
                                                                      time_atk[i]["end"],
                                                                      end_period, transition_i)

                if not impacted:
                    wrong_alert += 1
                    for v in alert_var:
                        false_alert_vars.append(v)
                    if fp_start is not None:
                        ifpt.append((alert - fp_start).total_seconds())
                    fp_start = alert

        start_p = time_atk[i]["start"]
        end_p = time_atk[i]["end"]

        if alert >= start_p and alert <= end_p + relaxed_win:
            if time_atk[i]["start"] not in detect_in_period:
                detect_in_period[time_atk[i]["start"]] = (alert - start_p).total_seconds()
                if is_invariant:
                    for t in time_atk[i]["attackType"]:
                        detect_type_repartition[t] += 1

            if not is_invariant:
                if time_atk[i]["start"] not in detect_period_atk_type:
                    for var in alert_var:
                        #FIXME var is (name, from, to) for timepattern
                        if var[0] in time_atk[i]["target"]:
                            for t in time_atk[i]["attackType"]:
                                detect_type_repartition[t] += 1

                            detect_period_atk_type[time_atk[i]["start"]] = (alert - start_p).total_seconds()
                            break

            for v in alert_var:
                true_alert_vars.append(v)

            true_positive += 1

    false_alert_repartition = counter_distribution(false_alert_vars)
    true_alert_repartition = counter_distribution(true_alert_vars)

    false_positive = wrong_alert
    false_negative = nbr_attack - true_positive
    true_negative = atk_trace_size - false_positive
    eval_res = evaluationIDS.EvalResult(true_positive, false_positive,
                                        true_negative, false_negative,
                                        nbr_attack)

    miss_atk = len(time_atk) - len(detect_in_period)
    return IDSResultData(detect_in_period, wrong_alert, miss_atk, ifpt, eval_res,
                         atk_types, detect_type_repartition,
                         false_alert_repartition, true_alert_repartition)

def get_stats(values):
    if len(values) == 0:
        return None, None, None, None
    else:
        return np.mean(values), np.std(values), np.min(values), np.max(values)

def display_repartition(atk_types, detect_types):
    s = "{"
    for i, x in enumerate(atk_types):
        if i != len(atk_types)-1:
            s += "{}:{}/{},".format(x, detect_types[x], atk_types[x])
        else:
            s += "{}:{}/{}".format(x, detect_types[x], atk_types[x])
    s += " }"
    return s


def run_ids_eval(func_ids, name, atk_time, atk_trace_size, aftereffect, window,
                 is_invariant, is_timepattern, result_file, count_trans, *args):

    ids = func_ids(*args)
    if is_timepattern:
        unique_transitions = None
        if count_trans:
            map_transition = get_first_transition(atk_time, ids.transition_log)
            unique_transitions = unique_impact_transitions(ids.transition_log, map_transition)
        ids_result = get_ids_result(ids, atk_time, atk_trace_size,
                                    unique_transitions, aftereffect, window,
                                    is_invariant, is_timepattern)
    else:
        ids_result = get_ids_result(ids, atk_time, atk_trace_size, None,
                                    aftereffect, window, is_invariant, is_timepattern)

    result_file.write("False Positive repartition: {}\n".format(ids_result.false_alert_repartition))
    result_file.write("True Positive repartition: {}\n".format(ids_result.true_alert_repartition))

    detected_period = set(ids_result.detected.keys())
    result_file.write("Detected Period\n")
    result_file.write(str(detected_period))
    result_file.write("Miss Period\n")
    all_period = set([period["start"] for period in atk_time])
    result_file.write(str(all_period - detected_period)+ "\n")

    result_file.write("Detection Time\n")
    detection_time = list(ids_result.detected.values())
    result_file.write(str(detection_time)+ "\n")
    mean_det, std_det, minval_det, maxval_det = get_stats(detection_time)
    mean_ifpt, std_ifpt, minval_ifpt, maxval_ifpt = get_stats(ids_result.inter_fp_time)

    result_file.write("{}\n".format(name))
    result_file.write("------------\n")
    result_file.write("Detected: {}\n".format(len(ids_result.detected)))
    result_file.write("Nbr Alert: {}\n".format(len(ids.malicious_activities)))
    result_file.write("Wrong alert: {}\n".format(ids_result.wrong_alert))
    result_file.write("Miss Attack: {}\n".format(ids_result.miss_atk))
    result_file.write("Per second {}\n".format(ids_result.eval_res))
    result_file.write("Detection Time Mean/Std/Min/Max: {}/{}/{}/{}\n".format(mean_det, std_det,
                                                                minval_det, maxval_det))

    result_file.write("Inter FP Time Mean/Std/Min/Max: {}/{}/{}/{}\n".format(mean_ifpt, std_ifpt,
                                                                             minval_ifpt, maxval_ifpt))
    result_file.write("Attack Type Repartition: {}\n".format(display_repartition(ids_result.atk_types, ids_result.detect_types)))
    result_file.write("Elapsed time for 1H: {}\n".format(ids.elapsed_time_per_computation))
    result_file.write("------------\n")

def main(inputfile, attackfile, conf, conf_inv, atk_time_file, run_ids,
         matrix, write, pred_file, map_id_pred, inv_file, pvstore_file, create_pv,
         result_filename, strategy, tp_filename, count_trans, trans_file):

    pv_store, data, data_atk = setup(inputfile, attackfile, conf, pvstore_file, create_pv)
    atk_file = evaluationIDS.create_expected_malicious_activities(atk_time_file)

    # window normally at 60

    with open(result_filename, "w") as result_file:

        if TIMEPAT in run_ids:
            run_ids_eval(run_time_pattern_ids, TIMEPAT, atk_file, len(data_atk), 10,
                         0, False, True, result_file, count_trans,
                         pv_store, data, data_atk, matrix, write,
                         strategy, tp_filename, trans_file)

        if AR in run_ids:
            run_ids_eval(run_ar_ids, AR, atk_file, len(data_atk), 10,
                         0, False, False, result_file, pv_store, data, data_atk)


        if INV in run_ids:
            pv_store_inv = pvStore.PVStore(conf_inv, data)

            run_ids_eval(run_invariant_ids, INV, atk_file, len(data_atk), 10,
                         0, True, False, result_file, pv_store_inv, data_atk, pred_file,
                         map_id_pred, inv_file)
                         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf")
    parser.add_argument("--confinv", action="store", dest="conf_inv")
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
    parser.add_argument("--create_pv", action="store_true", dest="create_pv", default=False)
    parser.add_argument("--pvstore", action="store", dest="pv_store_file")
    parser.add_argument("--result", action="store", dest="result_filename")
    parser.add_argument("--strategy", action="store", choices=["lof", "svm", "isolation"], default="lof",
                        dest="strategy")
    parser.add_argument("--filename", action="store", dest="tp_filename")
    parser.add_argument("--transition", action="store_true", dest="count_trans",default=False)
    parser.add_argument("--transfile", action="store", dest="trans_file")

    args = parser.parse_args()

    #FIXME configuration file for parameters

    main(args.input_file, args.attack_file, args.conf, args.conf_inv,
         args.atk_time, args.runids, args.matrix, args.write, args.pred_file,
         args.map_id_pred, args.inv_file, args.pv_store_file, args.create_pv,
         args.result_filename, args.strategy, args.tp_filename, args.count_trans,
         args.trans_file)
