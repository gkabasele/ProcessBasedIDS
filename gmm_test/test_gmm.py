from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import numpy as np
import pdb

def compute_denominator(value, model):
    denom = 0
    
    for i in range(3):
        denom += model.weights_[i] * stats.norm.pdf(value, model.means_[i][0],
                                                    np.math.sqrt(model.covariances_[i][0][0]))

    return denom

def prob_membership(value, weight, mean, std, denom):
    return (weight*stats.norm.pdf(value, mean, std))/denom

def test_membership(model, index, number_value):
    values = np.random.normal(model.means_[index][0],
                              np.math.sqrt(model.covariances_[index][0][0]),
                              number_value)
    found_index = None
    success = 0
    miss = 0
    for val in values:
        sample = np.array([val]).reshape(-1, 1)
        membership_proba = model.predict_proba(sample)
        found_index = np.where(membership_proba[0] == max(membership_proba[0]))[0][0]
        denom = compute_denominator(sample, model)
        all_member_proba = [prob_membership(sample, model.weights_[j], model.means_[j][0], np.math.sqrt(model.covariances_[j][0][0]), denom) for j in range(3)]
        expected_index = all_member_proba.index(max(all_member_proba))
        assert found_index == expected_index
        if found_index == index:
            success += 1
        else:
            miss += 1
    return success/number_value, miss/number_value

n = 10000 # number of sample

mus = [-1, 0, 3]
sigmas = [1.5**2, 1 **2, 0.5 **2]
dist_index = [0, 1, 2]
weights = [0.3, 0.5, 0.2]

samples = []

for i in range(n):
    z = np.random.choice(dist_index, p=weights)
    samples.append(np.random.normal(mus[z], sigmas[z], 1))

model = GaussianMixture(3).fit(samples)


for i in range(3):
    x, y = test_membership(model, i, 100)
    print("Index : {}\n".format(i))
    print("Success rate : {}%\n".format(x))
    print("Miss rate : {}%\n".format(y))
