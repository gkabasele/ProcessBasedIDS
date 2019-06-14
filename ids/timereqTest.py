import math
from datetime import datetime, timedelta
from timeChecker import TransitionMatrix
from utils import *

start_date = datetime.strptime("2018-12-25 15:10:00","%Y-%m-%d %H:%M:%S")

def test_transition_matrix():
    pv = ProcessVariable('127.0.0.1', 5020, HOL_REG, 0, limit_values=[0, 20, 40])
    t = TransitionMatrix(pv)
    val = [0, 5, 20, 35, 40]
    acc = 0
    for i, v in enumerate(val):
        acc = i * 5
        t.add_value(v, start_date + timedelta(seconds=acc))

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

    res = [[0, 10, -1], [-1, 0, 10], [-1, -1, 0]]

    for i in range(3):
        for j in range(3):
            assert res[i][j] == t.transitions[i][j]

test_transition_matrix()

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
test_distance_matrix()
