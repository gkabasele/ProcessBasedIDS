from datetime import datetime
from collections import OrderedDict

from simpleIDSTest import get_ids_result

def test_get_ids_result():

    p1 = {"start": datetime(2020, 10, 22, 10, 0, 0), "end": datetime(2020, 10, 22, 10, 5, 0)}
    p2 = {"start": datetime(2020, 10, 22, 10, 10, 0), "end": datetime(2020, 10, 22, 10, 10, 30)}
    p3 = {"start": datetime(2020, 10, 22, 10, 15, 0), "end": datetime(2020, 10, 22, 10, 16, 0)}
    p = [p1, p2, p3]

    malicious = OrderedDict()

    # scenario 1
    malicious[datetime(2020, 10, 22, 9, 58, 0)] = False
    malicious[datetime(2020, 10, 22, 10, 0, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 1, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 2, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 2, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 4, 0)] = True
    detected, wrong, missed =  get_ids_result(malicious.keys(), p)

    assert detected == 1
    assert missed == 2
    assert wrong == 1

    malicious.clear()

    #scenario 2
    malicious[datetime(2020, 10, 22, 9, 58, 0)] = False
    malicious[datetime(2020, 10, 22, 9, 59, 0)] = False
    malicious[datetime(2020, 10, 22, 10, 0, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 1, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 2, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 5, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 7, 0)] = False
    malicious[datetime(2020, 10, 22, 10, 8, 0)] = False
    malicious[datetime(2020, 10, 22, 10, 10, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 10, 10)] = True

    detected, wrong, missed =  get_ids_result(malicious.keys(), p)

    assert detected == 2
    assert missed == 1
    assert wrong == 4

    malicious.clear()

    #scenario 3
    malicious[datetime(2020, 10, 22, 10, 0, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 5, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 15, 10)] = True

    detected, wrong, missed =  get_ids_result(malicious.keys(), p)

    assert detected == 2
    assert missed == 1
    assert wrong == 0

    malicious.clear()

    #scenario 4
    malicious[datetime(2020, 10, 22, 10, 15, 10)] = True
    malicious[datetime(2020, 10, 22, 10, 20, 10)] = False

    detected, wrong, missed = get_ids_result(malicious.keys(), p)

    assert detected == 1
    assert missed == 2
    assert wrong == 1

    malicious.clear()

    #scenario 5
    malicious[datetime(2020, 10, 22, 9, 0, 0)] = False
    malicious[datetime(2020, 10, 22, 10, 0, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 1, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 2, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 3, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 4, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 5, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 8, 0)] = False
    malicious[datetime(2020, 10, 22, 10, 9, 0)] = False
    malicious[datetime(2020, 10, 22, 10, 9, 30)] = False
    malicious[datetime(2020, 10, 22, 10, 10, 0)] = True
    malicious[datetime(2020, 10, 22, 10, 10, 15)] = True
    malicious[datetime(2020, 10, 22, 10, 10, 29)] = True
    malicious[datetime(2020, 10, 22, 10, 10, 40)] = False
    malicious[datetime(2020, 10, 22, 10, 13, 29)] = False

    malicious[datetime(2020, 10, 22, 10, 15, 10)] = True
    malicious[datetime(2020, 10, 22, 10, 15, 59)] = True
    malicious[datetime(2020, 10, 22, 10, 16, 10)] = False

    detected, wrong, missed = get_ids_result(malicious.keys(), p)

    assert detected == 3
    assert missed == 0
    assert wrong == 7

test_get_ids_result()
