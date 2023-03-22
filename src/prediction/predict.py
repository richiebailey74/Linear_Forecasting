import numpy as np


def generatePredictions(x, weights, order, window):
    offset = order
    X = None

    for ind in range(window):
        if ind + offset == window: break
        X = x[ind:ind + offset].reshape(1, -1) if ind == 0 else np.concatenate((X, x[ind:ind + offset].reshape(1, -1)),
                                                                               axis=0)

    preds = X @ weights

    return preds


def generatePredictionsSpeech(x, weights, order, window):
    offset = order
    end_tracker = window - 1
    X = None

    for ind in range(window):
        if ind + offset == window: break
        X = x[ind:ind + offset].reshape(1, -1) if ind == 0 else np.concatenate((X, x[ind:ind + offset].reshape(1, -1)),
                                                                               axis=0)

    preds = X @ weights

    return preds