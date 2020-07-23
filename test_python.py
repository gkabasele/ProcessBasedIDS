import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from ar import arsel


class ARpredictor(object):

    def __init__(self):
        self.coef = list()

    def train(self, data):
        a = arsel(data, False, True, "AIC")
        self.coef = a.AR[0]
    
    def predict(self, data):
        if len(data) != len(self.coef):
            raise ValueError("Coefficient and data vector of differen size")
        else:
            return np.dot(self.coef, data) 


def compute_coef(filename, order=5):
    with open(filename, "r") as f:
        data = [float(value) for value in f]
        rho, _ = sm.regression.yule_walker(data, order=order,
                                           method="mle")
        print("STATS")
        print(rho)

        arsel_res = arsel(data, False, True, "AIC")
        print("AR")
        print(arsel_res.AR)



compute_coef("../autoregressive_opensource/test0.dat")
compute_coef("../autoregressive_opensource/test1.dat")
compute_coef("../autoregressive_opensource/test3.dat", order=2)
compute_coef("../autoregressive_opensource/rhoe.dat", order=11)
