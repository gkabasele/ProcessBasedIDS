import pdb
import argparse
import pickle
import os
from datetime import datetime
from datetime import timedelta
from scipy.stats import f
from copy import deepcopy
from matplotlib import pyplot as plt

from welford import Welford
import utils
import pvStore

from autoreg import ARpredictor
from idsAR import IDSAR
from idsAR import MyQueue

def assert_equal(expected, got):

    try:
        assert expected == got
    except AssertionError:
        print("******************")
        print("AssertionError, expected:{} got:{}".format(expected, got))
        print("******************")

def setup(normal_file, attack_file, conf):
    data = utils.read_state_file(normal_file)
    pv_store = pvStore.PVStore(conf, data)
    data_atk = utils.read_state_file(attack_file)

    return data, pv_store, data_atk

def test_myQueue():
    print("Testing myQueue")
    queue = MyQueue(maxlen=512)

    assert_equal(512, queue.maxlen)

    assert_equal(0, len(queue))

    for i in range(30):
        queue.add(i)

    assert_equal(30, len(queue))
    assert_equal(0, queue.queue[29])
    assert_equal(29, queue.queue[0])

    for i in range(30, 512):
        queue.add(i)

    assert_equal(512, len(queue))
    assert_equal(0, queue.queue[511])
    assert_equal(29, queue.queue[482])
    assert_equal(511, queue.queue[0])

    queue.add(512)

    assert_equal(512, len(queue))

    assert_equal(512, queue.queue[0])
    assert_equal(1, queue.queue[511])

    for i in range(513, 542):
        queue.add(i)

    assert_equal(512, len(queue))

    assert_equal(30, queue.queue[511])
    assert_equal(541, queue.queue[0])


def fscore(df1, v1, df2, v2, debug=False, alpha=0.05):
    if debug:
        pass
        #pdb.set_trace()
    F = v1/v2

    # Two-tailed test so alpha is divided by two
    crit1 = f.ppf(1 - alpha/2, df1, df2)
    crit2 = f.ppf(alpha/2, df1, df2)
    bound = sorted([crit1, crit2]) 
    # If H0 is rejected, then an attack occured
    return F <= bound[0] or F >= bound[1]

def fscore_right_tailed(df1, v1, df2, v2, alpha=0.05):
    F = v1/v2
    crit1 = f.ppf(1-alpha, df1, df2)
    return F >= crit1, F, crit1

def fscore_right_tailed_anomaly_detection(model, pred_error):
    try:
        return fscore_right_tailed(pred_error.k - 1, pred_error.std**2,
                                   model.res_dist.k -1, model.res_dist.std**2)
    except ZeroDivisionError:
        return False, None, None

def test_fscore_aux():
    print("Test F-test Two-tailed")
    assert_equal(False, fscore(15, 2.09, 20, 1.10, False, 0.10))
    assert_equal(False, fscore(40, 109.63, 20, 65.99, False, 0.05))
    assert_equal(True, fscore(304, 7.8**2, 304, 5.86**2, False, 0.05))
    assert_equal(False, fscore(11, 0.78, 14, 0.42, False, 0.05))

def test_fscore_right_tail():
    print("Test F-test Right-tailed")
    assert_equal(True, fscore_right_tailed(15, 2.09, 20, 1.10, 0.10)[0])

def get_num_denum(res_model, pre_model):
    if res_model.residual_var > pre_model.std**2:
        return res_model.residual_var, res_model.deg_free, pre_model.std**2, pre_model.k - 1
    else:
        return pre_model.std**2, pre_model.k - 1, res_model.residual_var, res_model.deg_free

def fscore_anomaly_detection(model, pred_error):
    try:
        num, df1, denum, df2 = get_num_denum(model, pred_error)

        debug = (num != 0 and denum != 0)

        return fscore(df1, num, df2, denum, debug, 0.1)

    except ZeroDivisionError:
        return False, None, None

def test_pv_running_ids(pv, data, data_atk):
    train_data = [state[pv] for state in data]
    attack_data = [state[pv] for state in data_atk]

    test_ar_model(train_data, attack_data, len(attack_data), pv)

    
def test_ar_model_from_file(order):
    filename = "./google_stock.txt"
    filename_atk = "./google_stock_dev.txt"
    data = list()
    data_atk = list()
    nbr_atk = 0
    with open(filename, "r", encoding="utf-8-sig") as fname:
        for line in fname:
            if not line.startswith("#"):
                data.append(float(line))

    with open(filename_atk, "r", encoding="utf-8-sig") as fname:
        for line in fname:
            if not line.startswith("#"):
                data_atk.append(float(line))
            else:
                nbr_atk += 1

    test_ar_model(data, data_atk, nbr_atk, filename_atk, order)

def test_ar_model(data, data_atk, nbr_atk, label, order=512):
    
    model = ARpredictor()
    model.train(data, maxorder=order)
    model.make_predictions_from_test(data)

    pdb.set_trace()
    pred_error = Welford()
    errors = list()
    nbr_anomalies_fscore = 0
    nbr_anomalies_shewart = 0

    crit_ts = list()
    fscore_ts = list()

    shewart_params = Welford()

    window = MyQueue(maxlen=model.order())
    for i, val in enumerate(data_atk):
        shewart_params(val)
        if model.out_of_range(val, 3, shewart_params.mean, shewart_params.std):
                nbr_anomalies_shewart += 1

        if len(window) == model.order():
            predictions = model.predict(window.queue)
            error = predictions - val
            tmp = deepcopy(pred_error)
            pred_error(error)
            errors.append(error)
            is_outlier, Fvalue, crit = fscore_right_tailed_anomaly_detection(model, pred_error)

            fscore_ts.append(Fvalue)
            crit_ts.append(crit)

            if is_outlier:
                pred_error = tmp
                nbr_anomalies_fscore += 1

        window.add(val)
    
    print("Residual: {}".format(model.res_dist))
    print("Error: {}".format(pred_error))
    print("For {}, {}/{}/{} (fscore/shewart/#state)".format(label,
                                                     nbr_anomalies_fscore,
                                                     nbr_anomalies_shewart,
                                                     nbr_atk))
    plt.plot(fscore_ts)
    plt.plot(crit_ts)

    plt.show()

def running_ids(data, data_atk, pv_store):

    ids = IDSAR(pv_store, data, control_coef=6, alpha=0.05)
    file_predictor = "./test_ar_predictor.bin"
    filename = "./test_ar_malicious.bin"
    file_reason = "./test_ar_reason.bin"

    print("Training autoregressive IDS")
    ids.create_predictors()
    ids.train_predictors()

    ids.run_detection_mode(data_atk)
    with open(filename, "wb") as f:
        pickle.dump(ids.malicious_activities, f)

    with open(file_reason, "wb") as f:
        pickle.dump(ids.malicious_reason, f)

def main(normal_file, attack_file, conf):
    test_myQueue()
    test_fscore_right_tail()

    data, pv_store, data_atk = setup(normal_file, attack_file, conf)

    test_pv_running_ids("ait401", data, data_atk)
    test_pv_running_ids("lit101", data, data_atk)
    #test_ar_model_from_file(order=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", action="store", dest="normal_file")
    parser.add_argument("--attack", action="store", dest="attack_file")
    parser.add_argument("--conf", action="store", dest="conf")
    args = parser.parse_args()
    main(args.normal_file, args.attack_file, args.conf)
