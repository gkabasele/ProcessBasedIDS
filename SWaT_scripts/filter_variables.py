import pickle
import argparse

def main(inputfile, outputfile, varfile):

    variables = None
    with open(varfile, mode="r") as fi:
        for line in fi:
            variables = line.split(",")
            variables = [x.replace("\n", "") for x in variables]

    variables.append("timestamp")
    with open(inputfile, mode="rb") as fi:
        data = pickle.load(fi)
        states = []
        for state in data:
            curr = {x:state[x] for x in variables}
            states.append(curr)

        with open(outputfile, mode="wb") as fo:
            pickle.dump(states, fo)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", action="store",
                        help="file containing the process over time as a list of dictionnary")
    parser.add_argument("--output", dest="output", action="store",
                        help="file to output the filtered variables")
    parser.add_argument("--variables", dest="varfile", action="store",
                        help="file containing the variables to be kept")

    args = parser.parse_args()

    main(args.input, args.output, args.varfile)
