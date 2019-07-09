import math
from datetime import datetime, timedelta
from timeChecker import TransitionMatrix, TimeChecker
from utils import *
import numpy as np
import pdb

start_date = datetime.strptime("2018-12-25 15:10:00","%Y-%m-%d %H:%M:%S")

vars_store = "./process_variables_crit.yml"

mu = 10
sigma = 2

def test_matrix():

    datas = []
    round_val_sens_start = [120.439, 120.46, 179.46, 248.234, 325.46]
    round_val_act_start = [1, 0, 2, 2, 2]

    timestamp = datetime.strptime("2018-12-25 15:10:00", "%Y-%m-%d %H:%M:%S")
    try:
        assert len(round_val_act_start) == len(round_val_sens_start)
    except AssertionError:
        print("Sens: {}, Act: {}".format(len(round_val_sens_start), len(round_val_act_start)))

    for x, y in zip(round_val_sens_start, round_val_act_start):
        delta = np.random.normal(mu, sigma, 1)[0]
        timestamp = timestamp + timedelta(seconds=delta)
        data = {'lit101': x, 'mv101': y, 'timestamp': timestamp}
        datas.append(data)

    round_value_sens = [495.065, 504.055, 573.932, 746.27, 800.70, 806.369,
                        812.75, 817.44, 815.29, 802.42, 739.42, 567.743,
                        504.029, 493.81, 494.83]

    round_value_act = [2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]

    try:
        assert len(round_value_act) == len(round_value_sens)
    except AssertionError:
        print("Sens: {}, Act: {}".format(len(round_value_sens), len(round_value_act)))

    for _ in range(10):
        for x, y in zip(round_value_sens, round_value_act):
            delta = np.random.normal(mu, sigma, 1)[0]
            timestamp = timestamp + timedelta(seconds=delta)

            data = {'lit101': x, 'mv101': y, 'timestamp': timestamp}

            datas.append(data)

    time_checker = TimeChecker(vars_store, datas)
    time_checker.get_values_timestamp()
    pdb.set_trace()

    atk_value_sens = [493.93, 504.74, 598.07, 744.19, 800.93, 810.49, 809.12,
                      811.39, 813.49, 814.03, 812.04, 815.94, 814.84, 817.32,
                      799.53, 737.41, 541.93, 507.074, 493.93]

    atk_value_act = [2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    try:
        assert len(atk_value_sens) == len(atk_value_act)
    except AssertionError:
        print("Sens: {}, Act: {}".format(len(atk_value_sens), len(atk_value_act)))

    datas_atk = []
    timestamp = datetime.strptime("2018-12-25 15:25:10", "%Y-%m-%d %H:%M:%S")
    for x, y  in zip(atk_value_sens, atk_value_act):
        delta = np.random.normal(mu, sigma, 1)[0]
        timestamp = timestamp + timedelta(seconds=delta)
        data = {'lit101': x, 'mv101':y, 'timestamp': timestamp}
        datas_atk.append(data)

    time_checker.detection_store = datas_atk
    time_checker.detect_suspect_transition()

    pdb.set_trace()
    atk_value_sens = [493.93, 504.74, 598.07, 744.19, 800.93, 810.49, 809.12,
                      811.39, 813.49, 817.50, 820.94, 823.29, 835.35, 822.83,
                      819.48]

    atk_value_act = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]

    try:
        assert len(atk_value_sens) == len(atk_value_act)
    except AssertionError:
        print("Sens : {}, Act: {}".format(len(atk_value_sens), len(atk_value_act)))

    datas_atk = []
    for x, y in zip(atk_value_sens, atk_value_act):
        delta = np.random.normal(mu, sigma, 1)[0]
        timestamp = timestamp + timedelta(seconds=delta)
        data = {'lit101': x, 'mv101': y, 'timestamp': timestamp}
        datas_atk.append(data)
    time_checker.detection_store = datas_atk
    time_checker.detect_suspect_transition()

test_matrix()

def test_transition_matrix():
    pv = ProcessVariable('127.0.0.1', 5020, HOL_REG, 0, limit_values=[0, 20, 40])
    t = TransitionMatrix(pv)
    val = [0, 5, 20, 35, 40]
    acc = 0
    for i, v in enumerate(val):
        acc = i * 5
        t.add_value(v, start_date + timedelta(seconds=acc), pv)

    val1 = 30
    val2 = 32
    val3 = 42

    assert not t.is_diff_reading(val1, val2)

    assert t.is_diff_reading(val1, val3)

    assert len(t.historic_val) == 5

    prob = t.compute_change_prob()

    assert prob == 1.0

    val = [40, 35, 30]
    for i, v in enumerate(val):
        acc = i * 5
        t.add_value(v, start_date + timedelta(seconds=acc))

    prob = t.compute_change_prob()

    assert prob == 6/7

    t.update_transition_matrix()


def distance_matrix(a, b):
    acc = 0
    for i in range(len(a.header)):
        for j in range(len(b.header)):
            val1 = a.transitions[i][j]
            val2 = b.transitions[i][j]
            acc += (val1 - val2)**2
    return math.sqrt(acc)

def test_distance_matrix():
    pv = ProcessVariable('127.0.0.1', 5020, HOL_REG, 0, limit_values=[1, 2, 3])
    t = TransitionMatrix(pv)
    t.transitions[0][1] = 20
    t.transitions[0][2] = 40

    t.transitions[1][0] = 30
    t.transitions[1][2] = 30
    t.transitions[2][0] = 20
    t.transitions[2][1] = 15

    m = TransitionMatrix(pv)

    m.transitions[0][1] = 17
    m.transitions[0][2] = 43

    m.transitions[1][0] = 29
    m.transitions[1][2] = 32
    m.transitions[2][0] = 16
    m.transitions[2][1] = 17

    d = distance_matrix(t, m)

    assert d == math.sqrt(43)
