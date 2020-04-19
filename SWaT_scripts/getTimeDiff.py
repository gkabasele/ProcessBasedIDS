import argparse
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--input", action="store", dest="filename")

args = parser.parse_args()

time_format = "%d/%m/%Y:%H:%M:%S"

start_date = datetime.strptime("28/12/2015:10:00:00",
                               time_format)

with open(args.filename, "r") as fname:
    for line in fname:
        cur_time = datetime.strptime(line.strip(), time_format)
        print("Diff with: " + line)
        print((cur_time - start_date).total_seconds())
