import argparse
import yaml
from py4j.java_gateway import JavaGateway
import predicate as pred
from pvStore import PVStore
from utils import TS
from utils import read_state_file
import pdb

TRANSACTIONS = "transactions"
MINSUPPORT = "minsupport"
FREQITEMSETS = "freqItemSets"
INVARIANTS = "invariants"

class ItemSet(object):

    def __init__(self, predicates):

        self.predicates = predicates
        self.support = None


    def compute_support(self, predicates):
        pass

    def __str__(self):
        return str(self.predicates)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.predicates.__len__()

    def __iter__(self):
        return self.predicates.__iter__()

def sensor_predicates(sensor, value, predicates, satisfied_pred, mapping_id_pred):
    for i, p in enumerate(predicates[sensor][pred.GT]):
        if p.is_true(value):
            p.support += 1
            mapping_id_pred[p.id] = p
            satisfied_pred.append((sensor, pred.GT, i))
            break
    for i, p in enumerate(predicates[sensor][pred.LS]):
        if p.is_true(value):
            p.support += 1
            mapping_id_pred[p.id] = p
            satisfied_pred.append((sensor, pred.LS, i))
            break

def actuator_predicates(actuator, value, predicates, satisfied_pred, mapping_id_pred):
    pred_on = predicates[actuator][pred.ON][0]
    pred_off = predicates[actuator][pred.OFF][0]
    if pred_on.is_true(value):
        pred_on.support += 1
        mapping_id_pred[pred_on.id] = pred_on
        satisfied_pred.append((actuator, pred.ON, 0))
    elif pred_off.is_true(value):
        pred_off.support += 1
        mapping_id_pred[pred_off.id] = pred_off
        satisfied_pred.append((actuator, pred.OFF, 0))
    else:
        raise ValueError("Actuator Value is neither true or false")

def get_sastisfied_predicate(state, predicates, mapping_id_pred):
    satisfied_pred = []
    for varname, val in state.items():
        if varname != TS:
            if pred.ON in predicates[varname]:
                actuator_predicates(varname, val, predicates, satisfied_pred, mapping_id_pred)
            else:
                sensor_predicates(varname, val, predicates, satisfied_pred, mapping_id_pred)
    return ItemSet(satisfied_pred)

def get_transactions(states, predicates, mapping_id_pred):
    transactions = [get_sastisfied_predicate(state, predicates, mapping_id_pred) for state in states]
    return transactions

def export_files(outfile, transactions, supportfile, predicates,
                 mappingfile, mapping_id_pred):

    with open(outfile, "w") as fname:
        for transaction in transactions:
            for item in transaction:
                varname, cond, index = item
                p = predicates[varname][cond][index]
                fname.write("{} ".format(p.id))
            fname.write("\n")

    with open(supportfile, "w") as fname:
        for varname in predicates:
            for cond in predicates[varname]:
                for p in predicates[varname][cond]:
                    fname.write("{} {}\n".format(p.id, p.support))

    with open(mappingfile, "w") as fname:
        for pred_id, pred in mapping_id_pred.items():
            fname.write("{}:{}\n".format(pred_id, pred))

def main(conf, infile, outfile, supportfile, mappingfile,
         invariants, freqfile):
    data = read_state_file(infile)
    predicates = pred.generate_all_predicates(conf, data)
    mapping_id_pred = {}
    transactions = get_transactions(data, predicates, mapping_id_pred)
    export_files(outfile, transactions, supportfile, predicates, mappingfile,
                 mapping_id_pred)

    print("Export transaction to {}".format(outfile))
    print("Export Support  to {}".format(supportfile))
    print("Create Java Gateway")

    # Mining invariants
    gateway = JavaGateway()

    cfp = gateway.entry_point.getCFP()
    miner = gateway.entry_point.getMiner()

    print("Running CFPGrowth Algorithm, exporting to {}".format(freqfile))

    cfp.runAlgorithm(outfile, freqfile, supportfile)

    minsup = 0.1 * cfp.getDatabaseSize()

    filtered_output = "_filter_{}".format(minsup).join([freqfile[:-4], ".txt"])

    print("Filtering Frequent Itemset to {} (minsup: {})".format(filtered_output,
                                                                 minsup))
    miner.filterItemSets(freqfile, filtered_output, minsup)

    print("Mining invariants")
    miner.fillItemSet(filtered_output)
    miner.miningRules()
    miner.exportRule(invariants)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf",
                        help="File with the process variable")
    parser.add_argument("--infile", action="store", dest="infile",
                        help="states of the system over time")
    parser.add_argument("--params", action="store", dest="params",
                        help="File containing the parameter for CFP")
    parser.add_argument("--map", action="store", dest="map_pred",
                        help="File where to write the mapping id to predicate")

    args = parser.parse_args()
    with open(args.params, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.BaseLoader) 

    main(args.conf, args.infile, cfg[TRANSACTIONS], cfg[MINSUPPORT],
         args.map_pred, cfg[INVARIANTS], cfg[FREQITEMSETS])
