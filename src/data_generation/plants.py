import numpy as np
from scipy.stats import norm


def generatePlantOutput(x, noise_power):
    order = 10
    offset = order
    end_tracker = x.shape[0] - 1
    X = None

    for ind in range(x.shape[0]):
        if ind + offset == x.shape[0]: break
        X = x[ind:ind + offset].reshape(1, -1) if ind == 0 else np.concatenate((X, x[ind:ind + offset].reshape(1, -1)),
                                                                               axis=0)

    w_s = np.ones(order).reshape(-1, 1)

    noiseToAdd = norm.rvs(size=x.shape[0] - order, scale=noise_power).reshape(-1, 1)

    temp = X @ w_s

    targets = temp + noiseToAdd

    return targets