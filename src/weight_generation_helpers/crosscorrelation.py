import numpy as np


def generateCrossCorrVec(x, y, order, window):
    # basically same as above but is correlation between input and targets

    offset = order
    end_tracker = window - 1
    X = None

    for ind in range(window):
        if ind + offset == window: break
        X = x[ind:ind + offset].reshape(1, -1) if ind == 0 else np.concatenate((X, x[ind:ind + offset].reshape(1, -1)),
                                                                               axis=0)

    targets = y[0:window - order].reshape(-1, 1)

    P = X.T @ targets

    P = P / window

    return P


def generateCrossCorrVecSpeech(x, y, order, window):
    # basically same as above but is correlation between input and targets

    offset = order
    end_tracker = window - 1
    X = None

    for ind in range(window):
        if ind + offset == window: break
        X = x[ind:ind + offset].reshape(1, -1) if ind == 0 else np.concatenate((X, x[ind:ind + offset].reshape(1, -1)),
                                                                               axis=0)

    targets = y[0:window - order].reshape(-1, 1)

    P = X.T @ targets

    P = P / window

    return P
