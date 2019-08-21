import math
from datetime import datetime, timedelta
import pdb
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from utils import *
from timeChecker import TransitionMatrix, TimeChecker
from timePattern import TimePattern

start_date = datetime.strptime("2018-12-25 15:10:00","%Y-%m-%d %H:%M:%S")

vars_store = "./process_variables_limit.yml"

pv = ProcessSWaTVar("lit101", "hr", limit_values=[100, 700], min_val=94, max_val=703)

def test_setup():

    mu = 140
    sigma = 20

    tmp = np.random.normal(mu, sigma, 20)

    mu = 2000
    sigma = 120

    trans_time = np.concatenate((tmp, np.random.normal(mu, sigma, 100)))
    return trans_time

def test_time_pattern():
    print("Starting Test time pattern")

    time_pattern = TimePattern()

    times = test_setup()

    for x in times:
        time_pattern.update(x)

    time_pattern.create_clusters()

    assert len(time_pattern.clusters) == 2

    mu = 140
    sigma = 20
    t = np.random.normal(mu, sigma, 1)
    c = time_pattern.get_cluster(t)
    assert c == time_pattern.clusters[0]

    mu = 2000
    sigma = 120
    t = np.random.normal(mu, sigma, 1)
    c = time_pattern.get_cluster(t)
    assert c == time_pattern.clusters[1]

    mu = 600
    sigma = 19
    t = np.random.normal(mu, sigma, 1)
    c = time_pattern.get_cluster(t)
    assert c == time_pattern.clusters[0]

    print("Ending Test time pattern")

test_time_pattern()

def show_kde(data):
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 100)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    y_data = density(xs)

    n, bins, patches = plt.hist(data, 100, density=True)
    plt.plot(xs, y_data)
    plt.show()

def _test_transition_matrix(val1, val2, matrix, mu, sigma, res, nb_round, pv):

    for i in range(nb_round):
        try:
            t = np.random.normal(mu, sigma, 1)
            resp, _ = matrix.check_transition_time(val1, val2, t, pv)
            assert resp == res
        except AssertionError:
            print("AssertionError: {}".format(t))

def test_transition_matrix():

    print("Starting Transition Matrix test")

    trans_time = test_setup()
    matrix = TransitionMatrix(pv)

    timestamp = datetime.strptime("2018-12-25 15:10:00", "%Y-%m-%d %H:%M:%S")

    #fill matrices
    for i in trans_time:
        matrix.update_transition_matrix(100, timestamp, pv)
        timestamp = timestamp + timedelta(seconds=i)
        matrix.update_transition_matrix(700, timestamp, pv)
        timestamp = timestamp + timedelta(seconds=200)

    matrix.compute_clusters()

    mu = 140
    sigma = 20
    val = 0
    print("Normal mode")

    _test_transition_matrix(700, 100, matrix, mu, sigma,
                            TransitionMatrix.SAME, 5, pv)
    mu = 2000
    sigma = 120

    _test_transition_matrix(700, 100, matrix, mu, sigma,
                            TransitionMatrix.SAME, 5, pv)

    print("Attack mode")
    mu = 500
    sigma = 20

    _test_transition_matrix(700, 100, matrix, mu, sigma,
                            TransitionMatrix.DIFF, 5, pv)
    mu = 1200
    sigma = 100

    _test_transition_matrix(700, 100, matrix, mu, sigma,
                            TransitionMatrix.DIFF, 5, pv)

    print("End Transition Matrix test")
test_transition_matrix()

def test_matrix():

    mu = 10
    sigma = 2
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

    round_value_sens = [495.065, 504.055, 544.89, 573.932, 746.27, 800.70, 806.369,
                        812.75, 817.44, 815.29, 802.42, 739.42, 567.743,
                        547.295, 504.029, 493.81, 494.83]

    round_value_act = [2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]

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

    atk_value_sens = [493.93, 504.74, 543.43, 598.07, 744.19, 800.93, 810.49, 809.12,
                      811.39, 813.49, 814.03, 812.04, 815.94, 814.84, 817.32,
                      799.53, 737.41, 544.93, 507.074, 493.93]

    atk_value_act = [2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

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
    atk_value_sens = [493.93, 504.74, 547.02, 598.07, 744.19, 800.93, 810.49, 809.12,
                      811.39, 813.49, 817.50, 820.94, 823.29, 835.35, 822.83,
                      819.48]

    atk_value_act = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]

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

#test_matrix()
