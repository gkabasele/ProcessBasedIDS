import pickle
import argparse

def main(inputfile, outputfile, start, end):

    with open(inputfile, mode="rb") as fi:
        data = pickle.load(fi)
        with open(outputfile, mode="wb") as fo:
            if not (start is None or end is None):
                pickle.dump(data[start:end], fo)
            elif end is None:
                pickle.dump(data[start:], fo)
            elif start is None:
                pickle.dump(data[:end], fo)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", action="store",
                        help="file containing the process over time as a list of dictionnary pickled")
    parser.add_argument("--output", dest="output", action="store",
                        help="file to output the sliced part of the input")
    parser.add_argument("--start", type=int, dest="start", action="store", default=0,
                        help="start value of the slice")
    parser.add_argument("--end", type=int, dest="end", action="store",
                        help="end value of the slice")

    args = parser.parse_args()
    
    main(args.input, args.output, args.start, args.end)


