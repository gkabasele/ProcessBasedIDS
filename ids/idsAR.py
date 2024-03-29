#! /usr/bin/env python3

from collections import OrderedDict
from collections import deque
from collections import Counter
from datetime import datetime
from argparse import ArgumentParser
from timeit import default_timer as timer
import pickle
import pdb
from copy import deepcopy
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from autoreg import ARpredictor
from autoreg import plot_residuals_stats
from reqChecker import Checker
from welford import Welford
from pvStore import PVStore

import utils

OFFLIMIT = True

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

    def __init__(self, pv_store, store, control_coef=3, alpha=0.05, maxorder=512):
        #alpha : signifiance level
        #control_coef: Stewart control limit

        Checker.__init__(self, pv_store, store)


        self.map_pv_predictor = dict()

        self.malicious_activities = OrderedDict()
        self.malicious_reason = list()

        self.control_coef = control_coef
        self.alpha = alpha
        self.maxorder = maxorder
        self.elapsed_time_per_computation = list()

    def create_predictors(self):
        for name in self.vars.continuous_monitor_vars():
            self.map_pv_predictor[name] = ARpredictor()

    def train_predictors(self):
        map_pv_col = {x: list() for x in self.map_pv_predictor.keys()}
        for _, state in enumerate(self.store):
            for name, val in state.items():
                if name in map_pv_col:
                    map_pv_col[name].append(val)

        for k, data in map_pv_col.items():
            model = self.map_pv_predictor[k]
            model.train(data, maxorder=self.maxorder)
            # compute the residuals for the model
            model.make_predictions_from_test(data)

            mu = np.mean(data)
            std = np.std(data)
            model.upper_limit = mu + self.control_coef * std
            model.lower_limit = mu - self.control_coef * std

    def export_model(self, filename):
        with open(filename, "wb") as fname:
            pickle.dump(self.map_pv_predictor, fname)

    def import_model(self, filename):
        with open(filename, "rb") as fname:
            self.map_pv_predictor = pickle.load(fname)


    def run_detection_mode(self, detection_store, debug_file=None):

        map_pv_hist = {k: MyQueue(maxlen=v.order()) for k, v in self.map_pv_predictor.items()}
        map_pv_pred_err = {k: Welford() for k in self.map_pv_predictor.keys()}
        map_pv_shewart = {k:Welford() for k in self.map_pv_predictor.keys()}

        if debug_file is not None:
            pred_residuals = list()
            confidence_interval_pos = list()
            confidence_interval_neg = list()
        start_timer = timer()

        for i, state in enumerate(detection_store):
            ts = state["timestamp"]
            if debug_file is not None:
                debug_file.write("State nbr:{}\n".format(i))

            if i % 3600 == 0:
                end_timer = timer()
                time_to_compute = end_timer - start_timer
                print("Elapsed time: {}".format(time_to_compute))
                self.elapsed_time_per_computation.append(time_to_compute)
                print("IDSAR Starting state {} of {}".format(i, len(detection_store)))

            for name, val in state.items():
                if name not in self.map_pv_predictor:
                    continue

                shewart = map_pv_shewart[name]
                model = self.map_pv_predictor[name]
                data = map_pv_hist[name].queue
                shewart(val)

                #if model.out_of_range(val, self.control_coef, shewart.mean, shewart.std):
                if val > model.upper_limit or val < model.lower_limit:
                    self.add_malicious_activities(ts, name)
                    self.malicious_reason.append(OFFLIMIT)

                elif len(data) == model.order():
                    prediction = model.predict(data)
                    residual = prediction - val
                    tmp_pred_err = deepcopy(map_pv_pred_err[name])
                    map_pv_pred_err[name](residual)
                    if debug_file is not None:
                        pred_residuals.append(residual)
                        confidence_interval_neg.append(-6*model.res_dist.std)
                        confidence_interval_pos.append(6*model.res_dist.std)

                    #if self.fscore_anomaly_detection(name, map_pv_pred_err[name]):
                    if self.is_outlier(residual, model):
                        self.add_malicious_activities(ts, name)
                        self.malicious_reason.append(not OFFLIMIT)
                        map_pv_pred_err[name] = tmp_pred_err

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

    def get_vars_alerts_hist(self):
        alert_occurence = [pv for variables in self.malicious_activities.values() for pv in variables]
        c = Counter(alert_occurence)
        total = sum(c.values(), 0.0)
        for key in c:
            c[key] /= total
        return c

    def fscore(self, df1, v1, df2, v2, alpha=0.05):
        F = v1/v2

        # Two-tailed test so alpha is divided by two
        crit1 = f.ppf(1 - alpha/2, df1, df2)
        crit2 = f.ppf(alpha/2, df1, df2)
        bound = sorted([crit1, crit2]) 
        # If H0 is rejected, then an attack occured
        return F <= bound[0] or F >= bound[1]

    def fscore_anomaly_detection(self, name, pred_error):
        model = self.map_pv_predictor[name]
        try:
            num, df1, denum, df2 = self.get_num_denom(model, pred_error)

            return self.fscore(df1, num, df2, denum, self.alpha)

        except ZeroDivisionError:
            return False


    def fscore_right_tailed(self, df1, v1, df2, v2, alpha=0.05):
        F = v1/v2
        crit1 = f.ppf(1-alpha, df1, df2)
        return F >= crit1

    def fscore_right_tailed_anomaly_detection(self, model, pred_error):
        try:
            return self.fscore_right_tailed(pred_error.k - 1, pred_error.std**2,
                                            model.res_dist.k -1, model.res_dist.std**2, self.alpha)
        except ZeroDivisionError:
            return False


    def anomaly_detected(self, name, pred_err):

        model = self.map_pv_predictor[name]

        try:
            # H0: The variances of the residual during the training phase and the detection phase are the same
            # Ha: The variances are different
            # H0 is rejected if the F value is higher than the critical value
            return self.fscore_right_tailed_anomaly_detection(model, pred_err)

        except ZeroDivisionError:
            return False

    # Residual follows a normal distribution so we can use the z-score to
    # detect if one is an outliers or not

    def is_outlier(self, residual, model):
        z_score = (residual - model.res_dist.mean)/model.res_dist.std

        return z_score > 6 or z_score < -6

    def get_num_denom(self, res_model, pre_model):
        if res_model.residual_var > pre_model.std**2:
            return res_model.residual_var, res_model.deg_free, pre_model.std**2, pre_model.k - 1
        else:
            return pre_model.std**2, pre_model.k - 1, res_model.residual_var, res_model.deg_free 

    def export_detected_atk(self, filename):
        with open(filename, "w") as f:
            for i, item in enumerate(self.malicious_activities.items()):
                k, v = item
                off_limit_reason = self.malicious_reason[i]
                if off_limit_reason:
                    info = "Out of range"
                else:
                    info = "Prediction error"
                f.write("[{}][{}] : [Var:{}]\n".format(k, info, v))

def main(normal_file, attack_file, conf_file, mal_file):
    data_norm = utils.read_state_file(normal_file)
    data_mal = utils.read_state_file(attack_file)
    pv_store = PVStore(conf_file, data_norm)

    ids = IDSAR(pv_store, data_norm, alpha=0.02)
    ids.create_predictors()
    ids.train_predictors()
    ids.run_detection_mode(data_mal)
    ids.export_detected_atk(mal_file)


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
