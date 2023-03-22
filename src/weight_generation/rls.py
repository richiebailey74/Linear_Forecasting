import numpy as np


def construct_RLS_weights(x, y, order, alpha=.9999):
    # initialize values
    R_inv = 100 * np.eye(order) * x.std()
    W = np.zeros(order).reshape(-1, 1)

    for i in range(x.shape[0]):

        if order + i == y.shape[0]: break

        X = x[i:order + i].reshape(-1, 1)
        d = y[i].reshape(-1, 1)
        pred = W.T @ X
        err = d - pred

        Z_k = R_inv @ X
        q_k = (X.T @ Z_k)[0][0]
        v_k = 1 / (alpha + q_k)
        Z_k_norm = v_k * Z_k

        W = W + Z_k_norm * err
        R_inv = (1 / alpha) * (R_inv - (Z_k_norm @ Z_k.T))

    return W
