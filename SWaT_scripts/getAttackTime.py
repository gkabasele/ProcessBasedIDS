import os
import sys
import argparse
from datetime import datetime
import pdb
import operator


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ids
import ids.utils as utils

time_format = "%d/%m/%Y:%H:%M:%S"

comp_sym = { "=": operator.eq,
             ">": operator.gt,
             ">=": operator.ge,
             "<=": operator.le,
             "<": operator.lt,
             "!=": operator.ne
           }

class Comparator(object):

    def __init__(self, op, value):
        self.op = op
        self.value = value

    def is_true(self, val):
        return self.op(val, self.value)

    def __str__(self):
        return "{}{}".format(self.op, self.value)

    def __repr__(self):
        return str(self)


def find_prev_change(data, ts, i, varname, from_val, to_val):
    try:
        if varname in data[0]:
            # Check if the process variable is in its attack value
            # if it is, then we have to look at when did the change happened
            # as it is the start of the attack
            if data[i]["timestamp"] == ts and data[i][varname] in to_val:
                j = i
                while j >= 0 and data[j][varname] in to_val:
                    j -= 1

                if j >= 0:
                    return data[j+1]["timestamp"]
            else:
                pdb.set_trace()
                raise ValueError("Wrong timestamp or value")
        else:
            print("Unknown varname: " + varname)
    except IndexError:
        pdb.set_trace()


def find_next_change(data, ts, i, varname, from_val, to_val):
    try:
        if varname in data[0]:
            # Check if the process variable is stays in the value of the attack
            # if it is, we need to check when it will change as it will be the
            # end of the attack
            if data[i]["timestamp"] == ts and data[i][varname] in from_val:
                j = i
                while j < len(data) and data[j][varname] in from_val:
                    j += 1

                if j < len(data):
                    return data[j]["timestamp"]
            else:
                pdb.set_trace()
                raise ValueError("Wrong timestamp or value")
    except IndexError:
        pdb.set_trace()

def find_prev_change_c(data, ts, i, varname, from_comp, to_comp):
    try:
        if varname in data[0]:
            # Check if the process variable is in its attack value
            # if it is, then we have to look at when did the change happened
            # as it is the start of the attack

            if data[i]["timestamp"] == ts and to_comp.is_true(data[i][varname]):
                j = i
                while j >= 0 and to_comp.is_true(data[j][varname]):
                    j -= 1

                if j >= 0:
                    return data[j+1]["timestamp"]
            else:
                pdb.set_trace()
                raise ValueError("Wrong timestamp or value")
    except IndexError:
        pdb.set_trace()

def find_next_change_c(data, ts, i, varname, from_comp, to_comp):
    try:
        if varname in data[0]:
            # Check if the process variable is in its attack value
            # if it is, then we have to look at when it change back to normal
            # as it is the end of the attack

            if data[i]["timestamp"] == ts and from_comp.is_true(data[i][varname]):
                j = i
                while j >= 0 and from_comp.is_true(data[j][varname]):
                    j += 1

                if j < len(data):
                    return data[j]["timestamp"]
            else:
                pdb.set_trace()
                raise ValueError("Wrong timestamp or value")
    except IndexError:
        pdb.set_trace()

def find_prev_brutal_change_c(data, ts, i, varname):
    # Look for the point where there as big difference in the update of
    # the process variable

    try:
        if varname in data[0]:
            if data[i]["timestamp"] == ts:
                j = i
                diff = 0
                while j > 0  and diff > 50:
                    cur_val = data[i][varname]
                    diff = abs(cur_val - data[j-1][varname])

                if j >= 0:
                    return data[j+1]["timestamp"]
            else:
                pdb.set_trace()
                raise ValueError("Wrong timestamp or value")
    except IndexError:
        pdb.set_trace()

def get_comparator(s):
    comp, val = s.split(":")
    return Comparator(comp_sym[comp], float(val))

def main(infile, d_intimes, c_intimes, outtimes):

    data = utils.read_state_file(infile)
    start_attacks_ts = list()

    with open(d_intimes, "r") as fname:
        for line in fname:
            if line.startswith("#") or line.startswith(" "):
                continue
            print(line)
            ts_tmp, i_tmp, var, from_tmp, to_tmp, kind, atk_id = [x.replace("\n", "") for x in line.split(",")]
            ts = datetime.strptime(ts_tmp, time_format)
            i = int(i_tmp)

            if to_tmp == "ON":
                to_val = [2]
            else:
                to_val = [0, 1]

            if from_tmp == "ON":
                from_val = [2]
            else:
                from_val = [0, 1]

            if kind == "change":
                res = find_prev_change(data, ts, i, var, from_val, to_val)
                if res is not None:
                    start_attacks_ts.append((res, int(atk_id)))
            elif kind == "stay":
                res = find_next_change(data, ts, i, var, from_val, to_val)
                if res is not None:
                    start_attacks_ts.append((res, int(atk_id)))
            else:
                raise ValueError("Disc. Unknown kind:" + kind)

    with open(c_intimes, "r") as fname:
        for line in fname:
            if line.startswith("#") or line.startswith(" "):
                continue
            print(line)
            ts_tmp, i_tmp, var, from_comp_tmp, to_comp_tmp, kind, atk_id = [x.replace("\n", "") for x in line.split(",")]
            ts = datetime.strptime(ts_tmp, time_format)
            i = int(i_tmp)

            from_comp = get_comparator(from_comp_tmp)
            to_comp = get_comparator(to_comp_tmp)

            if kind == "norm" or kind == "any":
                res = find_prev_change_c(data, ts, i, var, from_comp, to_comp)
                if res is not None:
                    start_attacks_ts.append((res, int(atk_id)))

            elif kind == "normend" or kind == "anyend":
                res = find_next_change_c(data, ts, i, var, from_comp, to_comp)
                if res is not None:
                    start_attacks_ts.append((res, int(atk_id)))

            elif kind == "brutal":
                res = find_prev_brutal_change_c(data, ts, i, var)
                if res is not None:
                    start_attacks_ts.append((res, int(atk_id)))

            else:
                raise ValueError("Cont. Unknown kind:" + kind)

    with open(outtimes, "w") as fname:
        start_attacks_ts.sort(key=lambda tup: tup[1])
        for res in start_attacks_ts:
            fname.write("{},{}\n".format(res[0].strftime(time_format), res[1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store", dest="infile")
    parser.add_argument("--disc_times", action="store", dest="d_intimes")
    parser.add_argument("--cont_times", action="store", dest="c_intimes")
    parser.add_argument("--output", action="store", dest="outtimes")

    args = parser.parse_args()

    main(args.infile, args.d_intimes, args.c_intimes, args.outtimes)
