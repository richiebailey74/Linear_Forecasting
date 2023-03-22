import numpy as np
from src.weight_generation_helpers import constructU


def construct_APA4_weights(x, order, K, nu=.1, lam=.1, verbose=True):
    # K is a hyperparamter for the number of samples to extend to

    W = np.zeros(order).reshape(-1, 1)

    for ind in range(x.shape[0]):

        if order + ind == x.shape[0] - K: break

        # construct U and d, then using nu and lambda to generate next weights
        U = constructU(x[ind:], order, K)

        d = x[ind + order:ind + K + order].reshape(-1, 1)

        W = ((1 - nu) * W) + nu * (U @ np.linalg.inv((U.T @ U) + lam * np.eye(K)) @ d)

    return W
