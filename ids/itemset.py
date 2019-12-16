import argparse
import pdb
import yaml
from py4j.java_gateway import JavaGateway
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
            try:
                if pred.ON in predicates[varname]:
                    actuator_predicates(varname, val, predicates, satisfied_pred, mapping_id_pred)
                else:
                    sensor_predicates(varname, val, predicates, satisfied_pred, mapping_id_pred)
            except KeyError:
                # Variable must be ignored
                pass
    return ItemSet(satisfied_pred)

def get_transactions(states, predicates, mapping_id_pred):
    transactions = [get_sastisfied_predicate(state, predicates, mapping_id_pred) for state in states]
    return transactions

def export_files(outfile, transactions, supportfile, predicates,
                 mappingfile, mapping_id_pred, gamma, theta):

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
                        fname.write("{} {}\n".format(p.id,
                                                     max(int(gamma*p.support),
                                                     int(theta))))

    with open(mappingfile, "w") as fname:
        for pred_id, pred in mapping_id_pred.items():
            fname.write("{}:{}\n".format(pred_id, pred))

def main(conf, infile, outfile, supportfile, mappingfile,
         invariants, freqfile, closefile, minsup_ratio, gamma, theta,
         ids_input, do_mining, do_detection):

    if gamma < 0 and gamma > 1:
        raise ValueError("Gamma must be between 0 and 1")

    if theta < 0 and theta > gamma:
        raise ValueError("Theta must be between 0 and gamma")

    data = read_state_file(infile)
    predicates = pred.generate_all_predicates(conf, data)
    mapping_id_pred = {}
    transactions = get_transactions(data, predicates, mapping_id_pred)
    export_files(cfg[TRANSACTIONS], transactions, cfg[MINSUPPORT],
                 predicates, mappingfile, mapping_id_pred, gamma, theta)

    print("Export transaction to {}".format(cfg[TRANSACTIONS]))
    print("Export Support  to {}".format(cfg[MINSUPPORT]))

    if do_mining:
        print("Create Java Gateway")

        # Mining invariants
        gateway = JavaGateway()

        cfp = gateway.entry_point.getCFP()
        miner = gateway.entry_point.getMiner()

        print("Running CFPGrowth Algorithm, exporting to {}".format(cfg[FREQITEMSETS]))


        cfp.runAlgorithm(cfg[TRANSACTIONS], cfg[FREQITEMSETS], cfg[MINSUPPORT])

        print("Mining invariants")
        miner.exportCloseItemsets(cfg[CLOSE], miner.miningRules(cfg[FREQITEMSETS]))
        miner.exportRule(cfg[INVARIANTS])

    if do_detection:
        print("Running the ids")
        ids = IDSInvariant(mapping_id_pred, cfg[INVARIANTS], cfg[LOG])
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
    parser.add_argument("--minsup", action="store", type=float, default=0.1, dest="minsup")
    parser.add_argument("--gamma", action="store", type=float, default=1.0, dest="gamma")
    parser.add_argument("--theta", action="store", type=float, default=0.0, dest="theta")

    args = parser.parse_args()
    with open(args.params, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.BaseLoader)

    main(args.conf, args.infile, cfg[TRANSACTIONS], cfg[MINSUPPORT],
         args.map_pred, cfg[INVARIANTS], cfg[FREQITEMSETS], cfg[CLOSE],
         args.minsup, args.gamma, args.theta, args.ids_input,
         args.do_mining, args.do_detection)