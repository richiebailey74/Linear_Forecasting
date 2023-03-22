import numpy as np


def generateAutoCorrMat(x, order, window):
    # take x and construct X based on order and window size
    # order is the number of columns of the delays
    # window size is the number of rows in the delay matrix (or is how far back it's reached into the signal)
    # then do X.T @ X to get the R matrix (autocorrelation matrix)

    offset = order
    end_tracker = window - 1
    X = None

    for ind in range(window):
        if ind + offset == window: break
        X = x[ind:ind + offset].reshape(1, -1) if ind == 0 else np.concatenate((X, x[ind:ind + offset].reshape(1, -1)),
                                                                               axis=0)

    R = X.T @ X
    R = R / window

    return R


def generateAutoCorrMatSpeech(x, order, window):
    # take x and construct X based on order and window size
    # order is the number of columns of the delays
    # window size is the number of rows in the delay matrix (or is how far back it's reached into the signal)
    # then do X.T @ X to get the R matrix (autocorrelation matrix)

    offset = order
    end_tracker = window - 1
    X = None

    for ind in range(window):
        if ind + offset == window: break
        X = x[ind:ind + offset].reshape(1, -1) if ind == 0 else np.concatenate((X, x[ind:ind + offset].reshape(1, -1)),
                                                                               axis=0)

    R = X.T @ X
    R = R / window

    return R