import numpy as np


def constructU(x, order, K):
    U = None
    for shift in range(K):
        u = x[shift:shift + order].reshape(-1, 1)
        U = u if shift == 0 else np.concatenate((U, u), axis=1)

    return U