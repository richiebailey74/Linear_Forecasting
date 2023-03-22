import numpy as np


def construct_LMS_weights(x, y, lr, order, verbose=False):
    # initialize weights
    w = np.zeros(order).reshape(-1, 1)

    # need to basically construct a sliding window that is the size of the filter order over the input signal
    for i in range(x.shape[0]):

        if i + order == x.shape[0] - order: break

        curr_range = x[i:i + order]

        pred = curr_range.reshape(1, -1) @ w
        error = y[i] - pred

        update = (2 * lr * error * curr_range).reshape(-1, 1)

        w = w + update

    return w