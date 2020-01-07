import argparse
import pdb
import yaml
import time
from py4j.java_gateway import JavaGateway
import numpy as np
import predicate as pred
from pvStore import PVStore
from utils import TS
from utils import read_state_file
from idsInvariants import IDSInvariant

TRANSACTIONS = "transactions"
MINSUPPORT = "minsupport"
FREQITEMSETS = "freqItemSets"
INVARIANTS = "invariants"
OUTPUT_RES = "output"
LOG = "log"
CLOSE = "closeItemset"
MAPPINGFILE = "mappingfile"

class HelpCounter(object):

    __slots__ = ['index']

    def __init__(self):
        self.index = 0

    def increment(self):
        self.index += 1

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

def get_feature_sensors(state, sensors, sensor):
    return [state[k] for k in sensors if k != sensor]

def sensor_predicates(state, sensors, sensor, value, predicates, satisfied_pred, mapping_id_pred, stop):
    #FIXME ensure the order of the feature compare to the coefficient
    features = np.array(get_feature_sensors(state, sensors, sensor)).reshape(1, -1)
    for i, p in enumerate(predicates[sensor][pred.GT]):
        if p.is_true_model(value, features):
            p.support += 1
            mapping_id_pred[p.id] = p
            satisfied_pred.append((sensor, pred.GT, i))
            if stop:
                break
    for i, p in enumerate(predicates[sensor][pred.LS]):
        if p.is_true_model(value, features):
            p.support += 1
            mapping_id_pred[p.id] = p
            satisfied_pred.append((sensor, pred.LS, i))
            if stop:
                break

def actuator_predicates(actuator, value, predicates, satisfied_pred, mapping_id_pred):
    pred_on = predicates[actuator][pred.ON][0]
    pred_off = predicates[actuator][pred.OFF][0]
    if pred_on.is_true_value(value):
        pred_on.support += 1
        mapping_id_pred[pred_on.id] = pred_on
        satisfied_pred.append((actuator, pred.ON, 0))
    elif pred_off.is_true_value(value):
        pred_off.support += 1
        mapping_id_pred[pred_off.id] = pred_off
        satisfied_pred.append((actuator, pred.OFF, 0))
    elif value == pred.OFFZ:
        pred_off.support += 1
        mapping_id_pred[pred_off.id] = pred_off
        satisfied_pred.append((actuator, pred.OFF, 0))
    else:
        raise ValueError("Actuator Value is neither true or false")

def get_sastisfied_predicate(state, sensors, predicates, mapping_id_pred, stop, counter):
    if counter.index % 50000 == 0:
        print("Up to transactions: {}".format(counter.index))

    satisfied_pred = []
    for varname, val in state.items():
        if varname != TS:
            try:
                if pred.ON in predicates[varname]:
                    actuator_predicates(varname, val, predicates, satisfied_pred,
                                        mapping_id_pred)
                else:
                    sensor_predicates(state, sensors, varname, val, predicates,
                                      satisfied_pred, mapping_id_pred, stop)
            except KeyError:
                # Variable must be ignored
                pass
    counter.increment()
    return ItemSet(satisfied_pred)

def get_transactions(states, sensors, predicates, mapping_id_pred, stop):
    counter = HelpCounter()
    start_time = time.time()
    transactions = [get_sastisfied_predicate(state, sensors, predicates, mapping_id_pred, stop, counter) for state in states]
    duration = time.time() - start_time
    print("Duration (s): {}".format(duration))
    return transactions

def count_predicates(predicates, sensors, actuators):
    count = 0
    for act in sensors:
        try:
            count += len(predicates[act][pred.GT])
            count += len(predicates[act][pred.LS])
        except KeyError:
            pass

    for sens in actuators:
        try:
            count += len(predicates[sens][pred.ON])
            count += len(predicates[sens][pred.OFF])
        except KeyError:
            pass

    return count

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
                    if p.support > 0:
                        fname.write("{} {}\n".format(p.id, p.support))

    with open(mappingfile, "w") as fname:
        for pred_id, pred in mapping_id_pred.items():
            fname.write("{}:{}\n".format(pred_id, pred))

def main(conf, infile, outfile, supportfile, mappingfile,
         invariants, freqfile, closefile, minsup_ratio,
         ids_input, do_mining, do_detection, stop):

    data = read_state_file(infile)
    store = PVStore(conf)
    sensors = store.continuous_monitor_vars()
    actuators = store.discrete_monitor_vars()

    predicates = pred.generate_all_predicates(conf, data)
    count = count_predicates(predicates, sensors, actuators)
    print("Number of count: {}".format(count))

    mapping_id_pred = {}
    transactions = get_transactions(data, sensors, predicates, mapping_id_pred, stop)
    export_files(outfile, transactions, supportfile,
                 predicates, mappingfile, mapping_id_pred)

    print("Export transaction to {}".format(outfile))
    print("Export Support  to {}".format(supportfile))

    if do_mining:
        print("Create Java Gateway")

        # Mining invariants
        gateway = JavaGateway()

        cfp = gateway.entry_point.getCFP()
        miner = gateway.entry_point.getMiner()

        print("Running CFPGrowth Algorithm, exporting to {}".format(freqfile))

        cfp.runAlgorithm(outfile, freqfile, supportfile)

        print("Mining invariants")
        miner.exportCloseItemsets(closefile, miner.miningRules(freqfile))
        miner.exportRule(invariants)

    if do_detection:
        print("Running the ids")
        ids = IDSInvariant(mapping_id_pred, invariants, cfg[LOG])
        data = read_state_file(ids_input)
        with open(cfg[OUTPUT_RES], "w") as fname:
            for state in data:
                invalid = ids.valid_state(state)
                if invalid is not None:
                    fname.write("{}\n".format(str(invalid)))
                    fname.write("{}\n\n".format(str(state)))

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
    parser.add_argument("--ids", action="store", dest="ids_input",
                        help="File to run the ids on")
    parser.add_argument("--mine", action="store_true", dest="do_mining",
                        help="should the invariant be mined")
    parser.add_argument("--run", action="store_true", dest="do_detection",
                        help="run the IDS")
    parser.add_argument("--stop", action="store_true", dest="stop",
                        help="stop after a predicate is satisfied pred")
    parser.add_argument("--minsup", action="store", type=float, default=0.1, dest="minsup")

    args = parser.parse_args()
    with open(args.params, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.BaseLoader)

    main(args.conf, args.infile, cfg[TRANSACTIONS], cfg[MINSUPPORT],
         cfg[MAPPINGFILE], cfg[INVARIANTS], cfg[FREQITEMSETS], cfg[CLOSE],
         args.minsup, args.ids_input, args.do_mining, 
         args.do_detection, args.stop)
