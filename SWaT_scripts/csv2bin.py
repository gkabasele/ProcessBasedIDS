import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ids/')))
from queue import Queue
import argparse
import pickle 
from reader import CSVReader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, dest="input")
    parser.add_argument("--output", type=str, dest="output")
    args = parser.parse_args()

    if args.input:
        queue = Queue()
        reader = CSVReader(args.input, [queue])
        
        reader.start()

        reader.join()
        res = [] 
        while not queue.empty():
            state = queue.get()
            res.append(state)
        pickle.dump(res, open(args.output, "wb")) 
    else:
        raise ValueError("No file to read")
