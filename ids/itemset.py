import argparse
import predicate as pred
from pvStore import PVStore
from utils import TS
from utils import read_state_file
import pdb

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

def main(conf, infile, outfile, supportfile, mappingfile):
    data = read_state_file(infile)
    predicates = pred.generate_all_predicates(conf, data)
    mapping_id_pred = {}
    transactions = get_transactions(data, predicates, mapping_id_pred)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", action="store", dest="conf")
    parser.add_argument("--infile", action="store", dest="infile")
    parser.add_argument("--database", action="store", dest="database",
                        help="File where to output the transaction")
    parser.add_argument("--map", action="store", dest="map_pred",
                        help="File where to write the mapping id to predicate")
    parser.add_argument("--support", action="store", dest="support",
                        help="File where to output the support")

    args = parser.parse_args()
    main(args.conf, args.infile, args.database, args.support, args.map_pred)
