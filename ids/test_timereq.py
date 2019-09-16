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

def setup_test():

    mu = 140
    sigma = 20

    tmp = np.random.normal(mu, sigma, 20)

    mu = 2000
    sigma = 120

    trans_time = np.concatenate((tmp, np.random.normal(mu, sigma, 100)))
    return trans_time

def test_time_pattern():
    print("Starting Test time pattern")

    nb_test = 0
    nb_test_fail = 0
    nb_test_pass = 0

    time_pattern = TimePattern()

    times = setup_test()

    for x in times:
        time_pattern.update(x)

    time_pattern.create_clusters()

    try:
        nb_test += 1
        assert len(time_pattern.clusters) == 2
        nb_test_pass += 1
    except AssertionError:
        nb_test_fail += 1

    mu = 140
    sigma = 20
    t = np.random.normal(mu, sigma, 1)
    c = time_pattern.get_cluster(t)
    
    try:
        nb_test += 1
        assert c == time_pattern.clusters[0]
        nb_test_pass += 1
    except AssertionError:
        nb_test_fail += 1

    mu = 2000
    sigma = 120
    t = np.random.normal(mu, sigma, 1)
    c = time_pattern.get_cluster(t)
    try:
        nb_test += 1
        assert c == time_pattern.clusters[1]
        nb_test_pass += 1
    except AssertionError:
        nb_test_fail += 1

    mu = 600
    sigma = 19
    t = np.random.normal(mu, sigma, 1)
    c = time_pattern.get_cluster(t)

    try:
        nb_test += 1
        assert c == time_pattern.clusters[0]
        nb_test_pass += 1
    except AssertionError:
        nb_test_fail += 1

    print("Ending Test time pattern")
    print("#Tests: {}, Passed:{}({}%), Failed:{}({}%)\n".format(nb_test, 
                                                                nb_test_pass,
                                                                nb_test_pass/nb_test,
                                                                nb_test_fail,
                                                                nb_test_fail/nb_test))

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


def test_transition_matrix():

    nb_test = 0
    nb_test_pass = 0
    nb_test_fail = 0

    def _test_transition_matrix(val1, val2, matrix, mu, sigma, res, nb_round, pv,
                                nb_test, nb_test_pass, nb_test_fail):
        tmp_test = nb_test
        tmp_pass = nb_test_pass
        tmp_fail = nb_test_fail

        for i in range(nb_round):
            t = np.random.normal(mu, sigma, 1)
            resp, _ = matrix.check_transition_time(val1, val2, t, pv)
            try:
                tmp_test += 1
                assert resp == res
                tmp_pass += 1
            except AssertionError:
                tmp_fail += 1

        return tmp_test, tmp_pass, tmp_fail


    print("Starting Transition Matrix test")

    trans_time = setup_test()
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

    nb_test, nb_test_pass, nb_test_fail = _test_transition_matrix(700, 100, matrix, mu, sigma,
                                                                  TransitionMatrix.SAME, 5, pv,
                                                                  nb_test, nb_test_pass,
                                                                  nb_test_fail)
    mu = 2000
    sigma = 120

    nb_test, nb_test_pass, nb_test_fail = _test_transition_matrix(700, 100, matrix, mu, sigma,
                                                                  TransitionMatrix.SAME, 5, pv,
                                                                  nb_test, nb_test_pass,
                                                                  nb_test_fail)

    print("Attack mode")
    mu = 500
    sigma = 20

    nb_test, nb_test_pass, nb_test_fail = _test_transition_matrix(700, 100, matrix, mu, sigma,
                                                                  TransitionMatrix.DIFF, 5, pv,
                                                                  nb_test, nb_test_pass,
                                                                  nb_test_fail)
    mu = 1200
    sigma = 100

    nb_test, nb_test_pass, nb_test_fail = _test_transition_matrix(700, 100, matrix, mu, sigma,
                                                                  TransitionMatrix.DIFF, 5, pv,
                                                                  nb_test, nb_test_pass,
                                                                  nb_test_fail)

    print("End Transition Matrix test")
    print("#Tests: {}, Passed:{}({}%), Failed:{}({}%)\n".format(nb_test,
                                                                nb_test_pass,
                                                                nb_test_pass/nb_test,
                                                                nb_test_fail,
                                                                nb_test_fail/nb_test))
test_transition_matrix()
