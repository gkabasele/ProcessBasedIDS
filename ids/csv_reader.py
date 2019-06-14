import csv
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--filename", type=str, action="store", dest="filename")

args = parser.parse_args()

filename = args.filename

fields = []
rows = []

with open(filename, 'r') as csvfile:
    cvsreader = csv.reader(csvfile)

    next(cvsreader)

    name_var = next(cvsreader)
    print("Name: {}".format(name_var))

    value_var = next(cvsreader)
    print("Name: {}".format(value_var))
