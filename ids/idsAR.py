#! /usr/bin/env python3

from collections import OrderedDict
from collections import deque
from datetime import datetime
from argparse import ArgumentParser
import pdb
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from autoreg import ARpredictor
from autoreg import plot_residuals_stats
from reqChecker import Checker
from welford import Welford
import utils

class MyQueue(object):

    def __init__(self, maxlen=None):
        self.queue = []
        self.maxlen = maxlen
        

    def is_empty(self):
        return len(self.queue) == 0
    
    def add(self, val):
        self.queue.insert(0, val) 
        if len(self.queue) > self.maxlen:
            self.queue.pop()

    def __str__(self):
        return self.queue.__str__()

    def __repr__(self):
        return self.queue.__str__()

    def __len__(self):
        return self.queue.__len__()

    def __iter__(self):
        return self.queue.__iter__()

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

        map_pv_hist = {k: MyQueue(maxlen=v.order()) for k, v in self.map_pv_predictor.items()}
        map_pv_pred_err = {k: Welford() for k in self.map_pv_predictor.keys()}

        if debug_file is not None:
            pred_residuals = list()
            confidence_interval_pos = list()
            confidence_interval_neg = list()

        for i, state in enumerate(detection_store):
            ts = state["timestamp"]
            if debug_file is not None:
                debug_file.write("State nbr:{}\n".format(i))

            for name, val in state.items():
                if name not in self.map_pv_predictor:
                    continue

                model = self.map_pv_predictor[name]
                data = map_pv_hist[name].queue

                if len(data) == model.order():
                    prediction = model.predict(data)
                    residual = val - prediction
                    map_pv_pred_err[name](residual)
                    if debug_file is not None:
                        pred_residuals.append(residual)
                        confidence_interval_neg.append(-6*model.res_dist.std)
                        confidence_interval_pos.append(6*model.res_dist.std)

                    #if self.anomaly_detected(name, val, map_pv_pred_err, differences_f):
                    if self.is_outlier(val, residual, model):
                        self.add_malicious_activities(ts, name)
                    else:
                        model.res_dist(residual)

                map_pv_hist[name].add(val)


        if debug_file is not None:
            plt.plot(pred_residuals)
            plt.plot(confidence_interval_pos, color="yellow")
            plt.plot(confidence_interval_neg, color="yellow")
            plt.show()
            plt.show()
            plt.hist(residual, density=True)
            plt.show()
            pdb.set_trace()

    def add_malicious_activities(self, ts, name):

        time_key = datetime(ts.year, ts.month, ts.day,
                            ts.hour, ts.minute, ts.second)
        if time_key not in self.malicious_activities:
            self.malicious_activities[time_key] = set()

        self.malicious_activities[time_key].add(name)

    def anomaly_detected(self, name, val, pred_err, differences_f):

        model = self.map_pv_predictor[name]
        if val > model.upper_limit or val < model.lower_limit:
            return True

        try:
            num, df1, denom, df2 = self.get_num_denom(model, pred_err[name])
            F = num/denom

            # H0: The variances of the residual during the training phase and the detection phase are the same
            # Ha: The variances are different
            # H0 is rejected if the F value is higher than the critical value

            # Two-tailed test so alpha is divided by two
            crit = f.ppf(1 - self.alpha/2, df1, df2)
            if F > crit:
                differences_f.append(F-crit)
                return True
        except ZeroDivisionError:
            return False

    # Residual follows a normal distribution so we can use the z-score to
    # detect if one is an outliers or not

    def is_outlier(self, val, residual, model):
        if val > model.upper_limit or val < model.lower_limit:
            return True

        z_score = (residual - model.res_dist.mean)/model.res_dist.std

        return z_score > 6 or z_score < -6

    def get_num_denom(self, res_model, pre_model):
        if res_model.residual_var > pre_model.std**2:
            return res_model.residual_var, res_model.deg_free, pre_model.std**2, pre_model.k - 1
        else:
            return pre_model.std**2, pre_model.k - 1, res_model.residual_var, res_model.deg_free 



def main(normal_file, attack_file, conf_file, mal_file):
    data_norm = utils.read_state_file(normal_file)
    data_mal = utils.read_state_file(attack_file)
    ids = IDSAR(conf_file, data_norm, alpha=0.02)
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
