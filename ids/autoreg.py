import pdb
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
from ar import arsel

from welford import Welford

class ARpredictor(object):

    def __init__(self):
        self.coef = list()
        self.residual_mean = None
        self.residual_var = None
        self.upper_limit = None
        self.lower_limit = None
        self.def_free = None

    def train(self, data):
        # arsel(data, submean=True, absrho=True, criterion=CIC, minorder=0, maxorder=512)
        a = arsel(data, False, True, "AIC")
        self.coef = [-x for x in a.AR[0]]

    def order(self):
        return len(self.coef) - 1

    def predict(self, data):
        # epsilon value
        if len(data) != len(self.coef) - 1:
            print(self.coef)
            raise ValueError("Coef. ({})and data ({}) vector of different size".format(len(self.coef)-1, len(data)))
        else:
            return np.dot(self.coef[1:], data)

    def make_predictions_from_test(self, dataset):

        order = len(self.coef) - 1 
        i = 0
        j = order
        predictions = list()
        residuals = list()

        while j < len(dataset):
            value = dataset[i:j]
            pred = self.predict(value[::-1])
            residuals.append(dataset[j] - pred)
            predictions.append(pred)
            i += 1
            j += 1

        self.deg_free = len(residuals) - 1

        plt.plot(range(len(residuals)), residuals)
        plt.show()

        # the residuals should not appear as autocorrelated, they should be
        # white noise

        plot_acf(residuals, lags=90, fft=True)
        plt.title("Residual AC")
        plt.show()

        plt.hist(residuals, density=True, bins=30)
        plt.title("Histogram residuals")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 301)
        kde = st.gaussian_kde(residuals)
        plt.plot(kde_xs, kde.pdf(kde_xs))
        plt.show()

        self.residual_mean = np.mean(residuals)
        self.residual_var = np.var(residuals)

        return predictions


def compute_coef_and_predict(filename, split_index):
    with open(filename, "r") as f:
        data = [float(value) for value in f]

        train, test = data[:split_index], data[split_index+1:]
        #train, test = data, data
        # train should be autocorrelated
        plot_acf(train, lags=90, fft=True)
        plt.show()

        model = ARpredictor()
        model.train(train)
        pdb.set_trace()
        predictions = model.make_predictions_from_test(test)

        mu_o = np.mean(test)
        mu_p = np.mean(predictions)

        std_o = np.std(test)

        upper_limit = mu_o + 3*std_o
        lower_limit = mu_o - 3*std_o

        print("mu_o:{}, mu_p:{}".format(mu_o, mu_p))
        print("var_o:{}, var_p:{}".format(np.var(test), np.var(predictions)))
        print("residual mean:{}".format(model.residual_mean))
        print("residual variance:{}".format(model.residual_var))

        plt.plot(test[len(model.coef)-1:])
        plt.plot(predictions, color='red')
        plt.axhline(y=upper_limit, color='orange', linestyle='-')
        plt.axhline(y=lower_limit, color='orange', linestyle='-')
        plt.show()

if __name__ == "__main__":
    #compute_coef_and_predict("../../autoregressive_opensource/test1.dat", 100)
    compute_coef_and_predict("../SWaT_scripts/lit301_normal.dat", 209700) # 419 400/209 700
    #compute_coef_and_predict("../SWaT_scripts/tank1_normal_1h.dat", 1000) # 300/2
