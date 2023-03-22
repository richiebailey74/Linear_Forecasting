import numpy as np
import matplotlib.pyplot as plt
from src.weight_generation import generateWienerWeights, generateWienerWeightsSpeech, construct_LMS_weights, construct_gamma_weights, construct_gamma_filter_weights_adaptive_mu
from src.weight_generation_helpers import constructU
from src.prediction import generatePredictions, generatePredictionsSpeech
from tabulate import tabulate


def compute_and_compare_NMSE(x,y,order_grid,window_grid):
    for i in order_grid:
        for j in window_grid:
            preds = generatePredictions(x, generateWienerWeights(x,y,i,j), i, j)
            targets = y[0:j-i].reshape(-1,1)
            mse = np.sum((preds - targets) ** 2) / j
            print("NMSE for filter order", i, "and window size of", j, ":", mse)
    return


def compute_and_compare_NMSE_voice(data,order_grid,window_grid):
    x = data
    for i in order_grid:
        y = data[1:]
        for j in window_grid:
            w = generateWienerWeightsSpeech(x,y,i,j)
            preds = generatePredictionsSpeech(x, w, i, j)
            targets = y[0:j-i].reshape(-1,1)
            mse = np.sum((preds - targets) ** 2) / np.sqrt(np.sum(data**2)) # squared error divided by the signal power
            print("NMSE for filter order", i, "and window size of", j, ":", mse)
    return


def compareWienerErrorOverTime(data, order, window):
    weights_time = None

    mse_vals = []
    for i in range(0, data.shape[0] - window, 10):
        y = data[i + 1:]
        x = data[i:]
        w = generateWienerWeights(x, y, order, window)
        preds = generatePredictions(x, w, order, window)
        targets = y[0:window - order].reshape(-1, 1)
        mse = np.sum((preds - targets) ** 2) / window
        mse_vals.append(mse)

    plt.plot(list(range(0, data.shape[0] - window, 10)), mse_vals)
    #     plt.plot(list(range(data.shape[0])), data)
    plt.title(f"Weiner Prediction Error in Time Compared to the\nInput Signal for Filter Order {order}")
    plt.xlabel("Signal index")
    plt.ylabel("Error")
    plt.show()


def construct_NLMS_weights_errorplot(x, y, lr, order):
    # initialize weights
    w = np.zeros(order).reshape(-1, 1)

    # need to basically construct a sliding window that is the size of the filter order over the input signal

    errs = []
    vals = []

    for i in range(x.shape[0]):

        if i + order == x.shape[0] - order: break

        curr_range = x[i:i + order]

        pred = curr_range.reshape(1, -1) @ w
        error = y[i] - pred
        eps = .00001  # to prevent ill-defined values (divide by 0)
        normalizer = np.sum((curr_range + eps) ** 2)
        errs.append(error ** 2)
        vals.append(i)

        update = (2 * lr * error * curr_range / normalizer).reshape(-1, 1)

        w = w + update

    errs = (np.array(errs) / len(vals)).ravel()
    plt.plot(vals, errs / 1000)
    plt.title(f"NMSE for NLMS Predictions for the Speech Signal of Order {order}")
    plt.xlabel("Index of Speech Signal")
    plt.ylabel("Error")
    plt.show()

    return w


def construct_RLS_weights_errorplot(x, y, order, alpha=.9999):
    # initialize values
    R_inv = 100 * np.eye(order) * x.std()
    W = np.zeros(order).reshape(-1, 1)

    errors = []

    for i in range(x.shape[0]):

        if order + i == y.shape[0]: break

        X = x[i:order + i].reshape(-1, 1)
        d = y[i + order - 1].reshape(-1, 1)
        pred = W.T @ X
        err = d - pred
        errors.append(err)

        Z_k = R_inv @ X
        q_k = (X.T @ Z_k)[0][0]
        v_k = 1 / (alpha + q_k)
        Z_k_norm = v_k * Z_k

        W = W + Z_k_norm * err
        R_inv = (1 / alpha) * (R_inv - (Z_k_norm @ Z_k_norm.T))

    plt.plot(np.array(list(range(y.shape[0] - order))), np.array(errors).ravel() ** 2 / x.shape[0])
    plt.xlabel("Signal Iteration Point")
    plt.ylabel("NMSE")
    plt.title(f"NMSE for RLS Prediction vs Signal Point for a\nForgetting Factor of {alpha} and Order of {order}")
    plt.show()

    return W


def computeNMSEValuesForRLSComparingAlpha(x, y):
    w_star = np.ones(10).reshape(-1, 1)
    orders = [6, 15]
    alphas = [.99, .999, .9999, .99999]

    for ords in orders:
        wsnr_vals = []
        for alp in alphas:
            w = construct_RLS_weights_errorplot(x, y, ords, alpha=alp)

    return


def construct_APA3_weights_errorplot(x, order, K, nu=.1, lam=.1, verbose=True):
    # K is a hyperparamter (I think) for the number of samples to extend to

    W = np.zeros(order).reshape(-1, 1)
    errs = []
    its = []

    for ind in range(x.shape[0]):

        if order + ind == x.shape[0] - K: break

        # construct U and d, then using nu and lambda to generate next weights
        U = constructU(x[ind:], order, K)

        d = x[ind + order:ind + K + order].reshape(-1, 1)

        errs.append(d[-1] - (U.T @ W)[-1])
        its.append(ind)

        W = ((1 - nu * lam) * W) + nu * (U @ (d - (U.T @ W)))

    if verbose:
        plt.plot(its, np.array(errs) ** 2 / len(its))
        plt.xlabel("Signal Iteration")
        plt.ylabel("NMSE")
        plt.title(f"APA3 NMSE Predictions for Filter Order {order}\nand for nu={nu} and lambda={lam}")

    return


def construct_APA4_weights_errorplot(x, order, K, nu=.1, lam=.1, verbose=True):
    # K is a hyperparamter (I think) for the number of samples to extend to

    W = np.zeros(order).reshape(-1, 1)
    errs = []
    its = []

    for ind in range(x.shape[0]):

        if order + ind == x.shape[0] - K: break

        # construct U and d, then using nu and lambda to generate next weights
        U = constructU(x[ind:], order, K)

        d = x[ind + order:ind + K + order].reshape(-1, 1)

        errs.append(d[-1] - (U.T @ W)[-1])
        its.append(ind)

        W = ((1 - nu) * W) + nu * (U @ np.linalg.inv((U.T @ U) + lam * np.eye(K)) @ d)

    if verbose:
        plt.plot(its, np.array(errs) ** 2 / len(its))
        plt.xlabel("Signal Iteration")
        plt.ylabel("NMSE")
        plt.title(f"APA4 NMSE Predictions for Filter Order {order}\nand for nu={nu} and lambda={lam}")

    return W, np.sum(np.array(errs) ** 2) / len(its)


def constructTableOfLMSPerformances(x, y):
    orders = list(range(25))

    data_to_tab = []

    for order in orders:
        w2, e2 = construct_LMS_weights(x / np.max(np.abs(x)), y / np.max(np.abs(y)), .0001, order)
        data_to_tab.append([order, e2 / np.max(np.abs(x))])

    col_names = ["Order", "NMSE"]

    print(tabulate(data_to_tab, headers=col_names))
    return


def constructTableOfGammaFilterPerformances(x, y):
    orders = list(range(1, 6))
    mus = list(range(1, 6, 1))
    mus = [x / 10 for x in mus]

    data_to_tab = []

    for order in orders:
        for mu in mus:
            weights, errors = construct_gamma_weights(x / np.max(np.abs(x)), y / np.max(np.abs(y)), order, .5,
                                                             mu)
            data_to_tab.append([order, mu, np.sum((errors ** 2) / (x.shape[0]))])

    column_names = ["Order", "Recursive Parameter", "NMSE"]

    print(tabulate(data_to_tab, headers=column_names))
    return


def constructTableOfGammaFilterPerformancesAdaptiveMu(x, y):
    orders = list(range(1, 6))
    mus = list(range(1, 6, 1))
    mus = [x / 10 for x in mus]

    data_to_tab = []

    for order in orders:
        for mu in mus:
            weights, errors = construct_gamma_filter_weights_adaptive_mu(x / np.max(np.abs(x)), y / np.max(np.abs(y)),
                                                                         order, .5, .001, mu)
            data_to_tab.append([order, mu, np.sum((errors ** 2) / (x.shape[0]))])

    column_names = ["Order", "Recursive Parameter Init", "NMSE"]

    print(tabulate(data_to_tab, headers=column_names))
    return




