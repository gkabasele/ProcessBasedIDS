import argparse
import yaml
import pdb

import utils

# Create yaml file with the attack time for the time pattern IDS
# Time are created from the attack binary file

def main(infile, outfile):

    data = utils.read_state_file(infile)
    timestamps = {}
    for state in data:
        if state["normal/attack"] == "Attack":
            timestamps[state["timestamp"]] = 1
        else:
            timestamps[state["timestamp"]] = 0

    with open(outfile, 'w') as fname:
        yaml.dump(timestamps, fname, default_flow_style=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", action="store", dest="infile")
    parser.add_argument("--outfile", action="store", dest="outfile")

    args = parser.parse_args()

    main(args.infile, args.outfile)
