import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--conf", type=str, dest="conf")
parser.add_argument("--infile", type=str, dest="infile")

args = parser.parse_args()

print(args.conf)
print(args.infile)
