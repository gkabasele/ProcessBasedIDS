import math
from datetime import datetime, timedelta
from timeChecker import TransitionMatrix, TimeChecker
from utils import *
import pdb

start_date = datetime.strptime("2018-12-25 15:10:00","%Y-%m-%d %H:%M:%S")

vars_store = "./process_variables_crit.yml"

def test_matrix():

    round_one = {
                    'lit101': 120.439,
                    'mv101': 1,
                    'timestamp': datetime.strptime("2018-12-25 15:10:00", "%Y-%m-%d %H:%M:%S")
                }

    round_two = {
                    'lit101': 120.46,
                    'mv101': 0,
                    'timestamp': datetime.strptime("2018-12-25 15:10:05", "%Y-%m-%d %H:%M:%S")
                }

    round_three = {
                    'lit101': 495.065,
                    'mv101': 2,
                    'timestamp': datetime.strptime("2018-12-25 15:10:10", "%Y-%m-%d %H:%M:%S")
                }

    round_four = {
                    'lit101': 504.055,
                    'mv101': 2,
                    'timestamp': datetime.strptime("2018-12-25 15:10:15", "%Y-%m-%d %H:%M:%S")
                }
    
    round_five = {
                    'lit101': 800.70,
                    'mv101':2,
                    'timestamp': datetime.strptime("2018-12-25 15:10:20", "%Y-%m-%d %H:%M:%S")
                }

    round_six = {
                    'lit101': 812.75,
                    'mv101':0,
                    'timestamp': datetime.strptime("2018-12-25 15:10:25", "%Y-%m-%d %H:%M:%S")
                }

    round_seven = {
                    'lit101': 817.44,
                    'mv101':0,
                    'timestamp': datetime.strptime("2018-12-25 15:10:30", "%Y-%m-%d %H:%M:%S")
                }

    data = [round_one, round_two, round_three, round_four, round_five, round_six, round_seven]

    time_checker = TimeChecker(vars_store, data)
    time_checker.get_values_timestamp()
    time_checker.compute_matrices()
    pdb.set_trace()

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
