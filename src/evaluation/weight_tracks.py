import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.weight_generation import generateWienerWeightsSpeech


def compareWienerWeightsOverTime(data, order, window):
    weights_time = None

    for i in range(0, data.shape[0] - window, 10):
        y = data[i + 1:]

        x = data[i:]
        w = generateWienerWeightsSpeech(x, y, order, window)
        weights_time = w if i == 0 else np.concatenate((weights_time, w), axis=1)

    frame = pd.DataFrame(weights_time.T,
                         columns=['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12',
                                  'w13', 'w14'])

    times = list(range(0, data.shape[0] - window, 10))
    plt.plot(times, frame)
    plt.title(f"Convergence of Weights: shifting window in time for order {order}")
    plt.show()
    return


def compareRLSWeightsOverTime(x, y, order, alpha):
    # initialize values
    R_inv = 100 * np.eye(order) * x.std()
    W = np.zeros(order).reshape(-1, 1)

    W_over_t = None

    for i in range(x.shape[0]):

        if order + i == y.shape[0]: break

        X = x[i:order + i].reshape(-1, 1)
        d = y[i + order - 1].reshape(-1,
                                     1)  # -1 is to align it (suppose I don't need to pass the y and can just get it from x)
        pred = W.T @ X
        err = d - pred

        Z_k = R_inv @ X
        q_k = (X.T @ Z_k)[0][0]
        v_k = 1 / (alpha + q_k)
        Z_k_norm = v_k * Z_k

        W = W + Z_k_norm * err
        R_inv = (1 / alpha) * (R_inv - (Z_k_norm @ Z_k_norm.T))

        W_over_t = W if i == 0 else np.concatenate((W_over_t, W), axis=1)

    # graph weights over time result
    frame = pd.DataFrame(W_over_t.T, columns=['w0', 'w1', 'w2', 'w3', 'w4', 'w5']) if order == 6 else pd.DataFrame(
        W_over_t.T,
        columns=['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14'])

    times = list(range(y.shape[0] - order))
    plt.plot(times, frame)
    plt.title(f"Convergence of Weights as RLS Weights are Computed\nfor Order {order} and Forgetting Factor {alpha}")
    plt.show()

    return


