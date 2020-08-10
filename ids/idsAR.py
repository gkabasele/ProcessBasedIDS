#! /usr/bin/env python3

from collections import OrderedDict
from collections import deque
from datetime import datetime
import numpy as np
from autoreg import ARpredictor
from reqChecker import Checker
from welford import Welford
from scipy.stats import f

class IDSAR(Checker):

    def __init__(self, descFile, store, detection_store, control_coef=3, alpha=0.05):
        #alpha : signifiance level
        #control_coef: Stewart control limit

        Checker.__init__(self, descFile, store)

        self.map_pv_predictor = dict()

        self.detection_store = None

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
                if name in self.map_pv_col:
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

    def run_detection_mode(self):
        if self.detection_store is None:
            raise(ValueError("No data to run the detection on"))

        map_pv_hist = {k: deque(maxlen=v.order()) for k, v in self.map_pv_predictor.items()}
        map_pv_pred_err = {k: Welford() for k in self.map_pv_predictor.keys()}

        for i, state in enumerate(self.detection_store):
            ts = state["timestamp"]
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

                map_pv_hist[name].add(val)

    def add_malicious_activities(self, ts, name):

        time_key = datetime(ts.year, ts.month, ts.day,
                            ts.hour, ts.minute, ts.second)
        if time_key not in self.malicious_activities:
            self.malicious_activities[time_key] = set()

        malicious_activities[time_key].add(name)

    def anomaly_detected(self, name, val, pred_err):

        model = self.map_pv_predictor[name]
        if val > model.upper_limit or val < model.lower_limit:
            return True

        F = model.residuals_var/(pred_err.std**2)
        df1 = model.deg_free
        df2 = pred_err.k - 1

        # H0: The variances of the residual during the training phase and the detection phase are the same
        # Ha: The variances are different

        crit1 = f.ppf(1-self.alpha/2, df1, df2)
        crit2 = f.ppf(self.alpha/2, df1, df2)

        return (F < crit1 or F > crit2)
