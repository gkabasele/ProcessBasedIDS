import argparse
import csv
import pickle
import pdb
from collections import OrderedDict
from datetime import datetime

"""
keep_var = ["fit101", "lit101", "mv101", "p101", "p102", "ait201","ait202", "ait203",
            "fit201", "mv201", "p201", "p202", "p203", "p204", "p205", "p206",
            "dpit301","fit301", "lit301", "mv301", "mv302", "mv302", "mv303",
            "mv304","p301", "p302", "ait401", "ait402", "fit401","lit401",
            "p401", "p402", "p403", "p404", "uv401", "ait501", "ait502",
            "ait503", "ait504", "fit501", "fit502", "fit503", "fit504", "p501",
            "p502", "pit501", "pit502", "pit503", "fit601", "p601", "p602",
            "p603", "timestamp"]
"""

def export_multiple_read(infile, outfile):
    states = []
    with open(infile, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            state = {}
            for x in row:
                key = x.lower().replace(" ", "")
                if key == 'timestamp':
                    ts = datetime.strptime(row[x], "%d/%m/%Y %I:%M:%S %p")
                    state[key] = ts
                elif key == 'normal/attack':
                    state[key] = row[x]
                else:
                    value = float(row[x].replace(",", "."))
                    state[key] = value
            states.append(state)
    with open(outfile, mode="wb") as bin_file:
        pickle.dump(states, bin_file)

def export_one_read(infile, outfile, keep_pv=None):

    with open(infile, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        column = OrderedDict((x.lower().replace(" ", ""), []) for x in csv_reader.fieldnames)
        for i, row in enumerate(csv_reader):
            for x in row:
                key = x.lower().replace(" ", "")
                if key == 'timestamp':
                    ts = datetime.strptime(row[x], " %d/%m/%Y %I:%M:%S %p")
                    column[key].append(ts)
                elif key == 'normal/attack':
                    column[key].append(row[x])
                else:
                    if keep_pv is None or key in keep_pv:
                        value = float(row[x].replace(",", "."))
                        column[key].append(value)
        pdb.set_trace()
    with open(outfile, mode="wb") as bin_file:
        pickle.dump(column, bin_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="infile", action="store", help="csv input file")
    parser.add_argument("--output", dest="outfile", action="store", help="binary output file")
    parser.add_argument("--keep", nargs="+", help="variables to keep in export",
                        dest="list_keep")

    args = parser.parse_args()
    export_one_read(args.infile, args.outfile, args.list_keep)
