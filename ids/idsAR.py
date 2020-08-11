#! /usr/bin/env python3

from collections import OrderedDict
from collections import deque
from datetime import datetime
from argparse import ArgumentParser
import pdb
import numpy as np
from scipy.stats import f
from autoreg import ARpredictor
from reqChecker import Checker
from welford import Welford
import utils

class IDSAR(Checker):

    def __init__(self, descFile, store, control_coef=3, alpha=0.05):
        #alpha : signifiance level
        #control_coef: Stewart control limit

        Checker.__init__(self, descFile, store)

        self.map_pv_predictor = dict()

        self.malicious_activities = OrderedDict()

        self.control_coef = control_coef
        self.alpha = alpha

    def create_predictors(self):

        for name, variable in self.vars.items():
            if not variable.is_bool_var():
                self.map_pv_predictor[name] = ARpredictor()

    def train_predictors(self):
        map_pv_col = {x: list() for x in self.map_pv_predictor.keys()}
        for _, state in enumerate(self.store):
            for name, val in state.items():
                if name in map_pv_col:
                    map_pv_col[name].append(val)

        for k, data in map_pv_col.items():
            model = self.map_pv_predictor[k]
            model.train(data)
            # compute the residuals for the model
            model.make_predictions_from_test(data)

            mu = np.mean(data)
            std = np.std(data)
            model.upper_limit = mu + self.control_coef * std
            model.lower_limit = mu - self.control_coef * std

    def run_detection_mode(self, detection_store, debug_file=None):

        map_pv_hist = {k: deque(maxlen=v.order()) for k, v in self.map_pv_predictor.items()}
        map_pv_pred_err = {k: Welford() for k in self.map_pv_predictor.keys()}

        for i, state in enumerate(detection_store):
            ts = state["timestamp"]
            if debug_file is not None:
                debug_file.write("State nbr:{}\n".format(i))

            for name, val in state.items():
                if name not in self.map_pv_predictor:
                    continue

                model = self.map_pv_predictor[name]
                data = map_pv_hist[name]

                if len(data) == model.order():
                    prediction = model.predict(data)
                    map_pv_pred_err[name](val - prediction)

                    if self.anomaly_detected(name, val, map_pv_pred_err):
                        self.add_malicious_activities(ts, name)

                map_pv_hist[name].append(val)

    def add_malicious_activities(self, ts, name):

        time_key = datetime(ts.year, ts.month, ts.day,
                            ts.hour, ts.minute, ts.second)
        if time_key not in self.malicious_activities:
            self.malicious_activities[time_key] = set()

        self.malicious_activities[time_key].add(name)

    def anomaly_detected(self, name, val, pred_err):

        model = self.map_pv_predictor[name]
        if val > model.upper_limit or val < model.lower_limit:
            pdb.set_trace()
            return True

        try:
            F = model.residual_var/(pred_err[name].std**2)
            df1 = model.deg_free
            df2 = pred_err[name].k - 1

            # H0: The variances of the residual during the training phase and the detection phase are the same
            # Ha: The variances are different

            crit1 = f.ppf(1-self.alpha/2, df1, df2)
            crit2 = f.ppf(self.alpha/2, df1, df2)

            if F < crit1 or F > crit2:
                pdb.set_trace()
                return True
            return False
        except ZeroDivisionError:
            return False

def main(normal_file, attack_file, conf_file, mal_file):
    data_norm = utils.read_state_file(normal_file)
    data_mal = utils.read_state_file(attack_file)
    pdb.set_trace()
    ids = IDSAR(conf_file, data_norm)
    ids.create_predictors()
    ids.train_predictors()
    pdb.set_trace()
    with open(mal_file, "w") as f:
        ids.run_detection_mode(data_mal)

        for k, v in ids.malicious_activities.items():
            f.write("{} : {}\n".format(k, v))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--normal", dest="normal_file", action="store",
                        help="file containing normal process")
    parser.add_argument("--attack", dest="attack_file", action="store",
                        help="file containing attack process")
    parser.add_argument("--malicious", dest="mal_file", action="store",
                        help="file where to export detection")
    parser.add_argument("--conf", dest="conf", action="store",
                        help="process variable conf file")
    args = parser.parse_args()

    main(args.normal_file, args.attack_file, args.conf, args.mal_file)
