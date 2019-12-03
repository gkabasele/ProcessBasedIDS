import argparse
import pickle
import pdb
import yaml
import utils

def read_data(inputname, outputname ,slen=utils.DAY_IN_SEC):

    variables = []

    with open(inputname, "rb") as fname:
        data = pickle.load(fname)

    diff_value = {k : set() for k in data[0].keys() if k != utils.TS and k != utils.CAT}
    min_value = {k : None for k in data[0].keys() if k != utils.TS and k != utils.CAT}
    max_value = {k : None for k in data[0].keys() if k != utils.TS and k != utils.CAT}

    for state in data[:slen]:
        for k, v in state.items():
            if k != utils.TS and k != utils.CAT:
                if len(diff_value[k]) <= 3:
                    diff_value[k].add(v)

                if min_value[k] is None:
                    min_value[k] = v
                else:
                    min_value[k] = min(v, min_value[k])

                if max_value[k] is None:
                    max_value[k] = v
                else:
                    max_value[k] = max(v, max_value[k])

    for k, v in diff_value.items():
        _type = utils.HOL_REG if len(v) > 3  else utils.DIS_COIL
        var = {"variable":
               {"name": k,
                "type": _type,
                "critical": [],
                "min": min_value[k],
                "max": max_value[k]
               }
              }
        variables.append(var)
    with open(outputname, "w") as ofh:
        content = yaml.dump(variables, default_flow_style=False)
        ofh.write(content)

def main(_input, output):

    read_data(_input, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", action="store")
    parser.add_argument("--output", dest="output", action="store")

    args = parser.parse_args()
    main(args.input, args.output)
