import numpy as np
from src.weight_generation import construct_APA3_weights, construct_APA4_weights


def optimizeHyperparametersForAPA3(x, order, K):
    nu_grid = [.00001, .0001, .001, .01, .1]

    lam_grid = [.00001, .0001, .001, .01, .1]

    nu_opt = None
    lam_opt = None
    min_err = np.inf

    for nu in nu_grid:

        for lam in lam_grid:

            w, e = construct_APA3_weights(x / np.max(x), 6, 10, nu=nu, lam=lam, verbose=False)
            if e < min_err:
                min_err = e
                nu_opt = nu
                lam_opt = lam

    print("Optimal nu:", nu_opt)
    print("Optimal lambda:", lam_opt)
    return


def optimizeHyperparametersForAPA4(x, order, K):
    nu_grid = [.00001, .0001, .001, .01, .1]

    lam_grid = [.00001, .0001, .001, .01, .1]

    nu_opt = None
    lam_opt = None
    min_err = np.inf

    for nu in nu_grid:

        for lam in lam_grid:

            w, e = construct_APA4_weights(x / np.max(x), 6, 10, nu=nu, lam=lam, verbose=False)
            if e < min_err:
                min_err = e
                nu_opt = nu
                lam_opt = lam

    print("Optimal nu:", nu_opt)
    print("Optimal lambda:", lam_opt)
    return


def optimizeSamplesValue(x):
    order = 6
    nu3 = .1
    lam3 = .00001
    nu4 = .01
    lam4 = .00001

    K_grid = [5, 10, 15, 20, 25, 50, 100]

    min_err3 = np.inf
    min_err4 = np.inf
    k3_opt = None
    k4_opt = None
    for k in K_grid:
        print("Starting value of k:", k)
        _, e3 = construct_APA3_weights(x / np.max(x), order, k, nu=nu3, lam=lam3, verbose=False)
        _, e4 = construct_APA4_weights(x / np.max(x), order, k, nu=nu4, lam=lam4, verbose=False)

        if e3 < min_err3:
            min_err3 = e3
            k3_opt = k

        if e4 < min_err4:
            min_err4 = e4
            k4_opt = k

    print("Optimal k value for APA3 is:", k3_opt)
    print("Optimal k value for APA4 is:", k4_opt)
