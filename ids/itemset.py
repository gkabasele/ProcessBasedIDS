import argparse
import pdb
import yaml
import time
import pickle
from py4j.java_gateway import JavaGateway
import numpy as np
import predicate as pred
from pvStore import PVStore
from utils import TS
from utils import read_state_file
from idsInvariants import IDSInvariant

TRANSACTIONS = "transactions"
TRANSBIN = "transactionsBin"
MINSUPPORT = "minsupport"
SUPPORT = "support"
FREQITEMSETS = "freqItemSets"
INVARIANTS = "invariants"
PREDICATE_EXPORT = "predicates"
OUTPUT_RES = "output"
LOG = "log"
CLOSE = "closeItemset"
MAPPINGFILE = "mappingfile"
MAPPINGFILEBIN = "mappingfileBin"

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

    # Get the GMM model of the deltas for the sensor
    # {Sensor : {Delta: GMM, DIST:[Predicate1, Predicate2]}}
    model = predicates[sensor][pred.DELTA]
    # Get the index of the valid predicate
    ind = model.get_valid_predicate(np.array(value).reshape(-1, 1))
    p = predicates[sensor][pred.DIST][ind]
    p.support += 1
    mapping_id_pred[p.id] = p
    satisfied_pred.append((sensor, pred.DIST, ind))

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
    for sensor in sensors:
        try:
            if pred.GT in predicates[sensor]:
                count += len(predicates[sensor][pred.GT])

            if pred.LS in predicates[sensor]:
                count += len(predicates[sensor][pred.LS])

            if pred.DIST in predicates[sensor]:
                count += len(predicates[sensor][pred.DIST])
        except KeyError as err:
            print(err)
            pdb.set_trace()
            print("KeyError")
        except TypeError as err:
            print(err)
            pdb.set_trace()
            print("TypeError")

    for actuator in actuators:
        try:
            count += len(predicates[actuator][pred.ON])
            count += len(predicates[actuator][pred.OFF])
        except KeyError:
            pass

    return count

def export_files(outfile, transactions, minsupportfile, supportfile, predicates,
                 mappingfile, mapping_id_pred, gamma, theta):


    with open(outfile, "w") as fname:
        for transaction in transactions:
            database = list()
            for item in transaction:
                varname, cond, index = item
                p = predicates[varname][cond][index]
                database.append(p.id)
            database.sort()
            for item in database:
                fname.write("{} ".format(item))
            fname.write("\n")

    mis_list = list()
    support_list = list()
    nbr_frequent = 0
    with open(minsupportfile, "w") as fname:
        for varname in predicates:
            for cond in predicates[varname]:
                if cond != pred.DELTA:
                    for p in predicates[varname][cond]:
                        mis_list.append((p.id, max(int(gamma*p.support), theta)))
                        support_list.append((p.id, p.support))
                        if p.support >= theta:
                            nbr_frequent += 1

        mis_list.sort(key=lambda x: x[0])
        support_list.sort(key=lambda x: x[0])
        for items in mis_list:
            itemsID, support = items
            fname.write("{} {}\n".format(itemsID, support))

    print("Number of items with a support greater than {} : {}".format(theta, nbr_frequent))
    with open(supportfile, "w") as fname:
        for items in support_list:
            itemsID, support = items
            fname.write("{} {}\n".format(itemsID, support))

    with open(mappingfile, "w") as fname:
        for pred_id, predicate in mapping_id_pred.items():
            fname.write("{}:{}\n".format(pred_id, predicate))

def find_mapping_pred_id(mapping_id_pred, transactions, predicates):
    for trans in transactions:
        for item in trans.predicates:
            varname, cond, index = item
            p = predicates[varname][cond][index]
            mapping_id_pred[p.id] = p

def main(conf, infile, outfile, minsupportfile, supportfile,
         mappingfile, invariants, freqfile, closefile,
         predicate_bin, transbin, mappingfilebin, do_transaction, do_predicate, ids_input, do_mining,
         do_detection, stop, gamma, theta):

    data = read_state_file(infile)
    store = PVStore(conf)
    sensors = store.continuous_monitor_vars()
    actuators = store.discrete_monitor_vars()
    predicates = None

    if do_predicate:
        predicates = pred.generate_all_predicates(conf, data)
        with open(predicate_bin, "wb") as f:
            pickle.dump(predicates, f)
    else:
        with open(predicate_bin, "rb") as f:
            predicates = pickle.load(f)

    count = count_predicates(predicates, sensors, actuators)
    print("Number of count: {}".format(count))

    mapping_id_pred = {}
    transactions = []

    if do_transaction:
        transactions = get_transactions(data, sensors, predicates, mapping_id_pred, stop)
        with open(transbin, "wb") as f:
            pickle.dump(transactions, f)
        with open(mappingfilebin, "wb") as f:
            pickle.dump(mapping_id_pred, f)
    else:
        with open(transbin, "rb") as f:
            transactions = pickle.load(f)

        try:
            with open(mappingfilebin, "rb") as f:
                mapping_id_pred = pickle.load(f)
        except EOFError as err:
            find_mapping_pred_id(mapping_id_pred, transactions, predicates)

        
    mis_theta = int(theta*len(data))
    export_files(outfile, transactions, minsupportfile, supportfile,
                 predicates, mappingfile, mapping_id_pred, gamma, mis_theta)

    print("Export transaction to {}".format(outfile))
    print("Export Support  to {}".format(supportfile))

    if do_mining:
        print("Create Java Gateway")

        # Mining invariants
        gateway = JavaGateway()

        cfp = gateway.entry_point.getCFP()
        miner = gateway.entry_point.getMiner()

        print("Running CFPGrowth Algorithm, exporting to {}".format(freqfile))

        cfp.runAlgorithm(outfile, freqfile, minsupportfile)

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
    parser.add_argument("--do_predicate", action="store_true", dest="do_predicate")
    parser.add_argument("--do_transaction", action="store_true", dest="do_transaction")
    parser.add_argument("--gamma", action="store", type=float, default=0.9, dest="gamma")
    parser.add_argument("--theta", action="store", type=float, default=0.32, dest="theta")

    args = parser.parse_args()
    with open(args.params, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.BaseLoader)

    main(args.conf, args.infile, cfg[TRANSACTIONS], cfg[MINSUPPORT], cfg[SUPPORT],
         cfg[MAPPINGFILE], cfg[INVARIANTS], cfg[FREQITEMSETS], cfg[CLOSE],
         cfg[PREDICATE_EXPORT], cfg[TRANSBIN], cfg[MAPPINGFILEBIN], 
         args.do_transaction, args.do_predicate, args.ids_input,
         args.do_mining, args.do_detection, args.stop,
         args.gamma, args.theta)
