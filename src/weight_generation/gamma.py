import numpy as np


# in this case, we can intuitively view the filter order as how many LC components into the past we would like to allow
def construct_gamma_weights(x, y, order, lr, mu):
    w = np.array([1 for x in range(order)]).reshape(-1, 1)

    # construct all x vectors (will add to as we iterate over the data)
    X = x.reshape(1, -1)

    for i in range(1, order):
        X = np.concatenate((X, np.zeros((1, x.shape[0]))), axis=0)

    # iterate over entire signal
    for i in range(x.shape[0]):

        temp_x = X[:, i].reshape(1, -1)

        pred = temp_x @ w

        e = y[i] - pred

        # fill in next column of X using recursive parameter (for part ii needs to be tested for .1,.2,.3,.4,.5)
        # update next col if not in last col to not step out of bounds
        if i != x.shape[0] - 1:
            # start at 1 not 0 bc first row of X is already computed
            # will update the i+1th column every signal iteration
            for j in range(1, order):
                X[j, i + 1] = X[j, i] * (1 - mu) + X[j - 1, i] * mu

        # update weights with online learning formulation
        temp_x = temp_x.reshape(-1, 1)
        w = w + 2 * e * lr * temp_x

    return w


# in this case, we can intuitively view the filter order as how many LC components into the past we would like to allow
def construct_gamma_filter_weights_adaptive_mu(x, y, order, lr1, lr2, mu_init):
    mu = mu_init

    w = np.array([1 for x in range(order)]).reshape(-1, 1)

    # construct all x vectors (will add to as we iterate over the data)
    X = x.reshape(1, -1)
    A = np.zeros((1, x.shape[0]))

    for i in range(1, order):
        X = np.concatenate((X, np.zeros((1, x.shape[0]))), axis=0)
        A = np.concatenate((A, np.zeros((1, x.shape[0]))), axis=0)

    errs = []

    # iterate over entire signal
    for i in range(x.shape[0]):

        temp_x = X[:, i].reshape(1, -1)

        pred = temp_x @ w

        e = y[i] - pred
        errs.append(e)

        # fill in next column of X using recursive parameter (for part iii needs to be initialized for .1,.2,.3,.4,.5)
        # fill in next column of A using X and recursive parameter
        # update next col if not in last col to not step out of bounds
        if i != x.shape[0] - 1:
            # start at 1 not 0 bc first row of X is already computed
            # will update the i+1th column every signal iteration
            for j in range(1, order):
                X[j, i + 1] = X[j, i] * (1 - mu) + X[j - 1, i] * mu
                A[j, i + 1] = (1 - mu) * A[j, i] + mu * A[j - 1, i] + X[j - 1, i] - X[j, i]

        # update the recursive parameter
        temp_a = A[:, i].reshape(1, -1)
        mu = mu + 2 * e * lr2 * (temp_a @ w)

        # update weights with online learning formulation
        temp_x = temp_x.reshape(-1, 1)
        w = w + 2 * e * lr1 * temp_x

    return w, np.array(errs)

