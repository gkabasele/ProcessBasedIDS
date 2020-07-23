import pdb
import matplotlib.pyplot as plt
import numpy as np
from ar import arsel

class ARpredictor(object):

    def __init__(self):
        self.coef = list()

    def train(self, data):
        a = arsel(data, False, True, "AIC")
        self.coef = [-x for x in a.AR[0]]

    def predict(self, data):
        # epsilon value
        if len(data) != len(self.coef) - 1:
            print(self.coef)
            raise ValueError("Coef. ({})and data ({}) vector of differen size".format(len(self.coef)-1, len(data)))
        else:
            return np.dot(self.coef[1:], data)

def compute_coef_and_predict(filename, order=5):
    with open(filename, "r") as f:
        data = [float(value) for value in f]
        if len(data) < 2000:
            train, test = data[:99], data[100:]
        else:
            train, test = data[:999], data[1000:]

        model = ARpredictor() 
        model.train(train)
        order = len(model.coef)

        i = 0
        j = order-1
        predictions = list()
        while j < len(test):
            value = test[i:j]
            pred = model.predict(value)
            predictions.append(pred)
            i += 1
            j += 1

        plt.plot(test[order-1:])
        plt.plot(predictions, color='red')
        plt.show()

compute_coef_and_predict("../autoregressive_opensource/test1.dat")
