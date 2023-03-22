import numpy as np
import matplotlib.pyplot as plt
from src.weight_generation import generateWienerWeights, construct_LMS_weights, construct_RLS_weights
from src.data_generation import generatePlantOutput


# reshapes the appropriate vector of weights to preserve optimal weights in all circumstances
def compute_wsnr(w_n, w_star):
    dif = np.abs(w_n.shape[0] - w_star.shape[0])
    if w_n.shape[0] > w_star.shape[0]:
        w_star = np.concatenate((w_star, np.zeros(dif).reshape(-1, 1)), axis=0)
    elif w_n.shape[0] < w_star.shape[0]:
        w_n = np.concatenate((w_n, np.zeros(dif).reshape(-1, 1)), axis=0)

    numer = w_star.T @ w_star
    denom = (w_star - w_n).T @ (w_star - w_n)
    return 10 * np.log(numer / denom)[0][0]


def compute_and_compare_WSNR(x, y, order_grid, window_grid):
    for i in order_grid:
        for j in window_grid:
            w_star = np.ones(10).reshape(-1,1)
            wsnr = compute_wsnr(generateWienerWeights(x,y,i,j),w_star)
            print("WSNR for filter order", i, "and window size of", j, ":", wsnr)
    return


def compareWindowSizes(x, noise, filterOrderGrid, windowSizes, w_star):
    y = generatePlantOutput(x, noise)
    for order in filterOrderGrid:
        wsnrVals = []
        for window in windowSizes:
            wsnrVals.append(compute_wsnr(generateWienerWeights(x,y,order,window),w_star))
        plt.plot(windowSizes,wsnrVals)
        plt.xlabel("Window Size")
        plt.ylabel("WSNR")
        plt.title(f"WSNR with Respect to Different Window Sizes Comparing\nWiener Filter Weights to Optimal Transfer Function Weights\nfor Filter Order of {order} and noise {noise}")
        plt.show()
    return


def compute_WSNR_for_LMS_across_learning_rates(x, y, lr_grid, order_grid, opt_ord=10):
    for j in order_grid:
        wsnrVals = []
        for i in lr_grid:
            w = construct_LMS_weights(x, y, i, j)
            w_star = np.ones(opt_ord).reshape(-1, 1) # for when optimal filter order is 10
            wsnr = compute_wsnr(w, w_star)
            wsnrVals.append(wsnr)

        plt.plot(lr_grid, wsnrVals)
        plt.xlabel("Learning Rate")
        plt.ylabel("WSNR")
        plt.title(
            f"WSNR with Respect to Different Learning Rates Comparing\nLMS Weights to Optimal Transfer Function Weights\nfor Filter Order of {j} and noise power {.1}")
        plt.show()
    return


def computeWSNRValueForRLS(x,y):
    w_star = np.ones(10).reshape(-1,1)
    orders = [5,10,15,30]
    for ords in orders:
        w = construct_RLS_weights(x,y,ords)
        wsnr = compute_wsnr(w,w_star)
        print(f"The WSNR value for RLS for the plant when compared to optimal weights for order {ords} is: {wsnr}")
    return


def computeWSNRValuesForRLSComparingAlpha(x, y, noise=.1):
    w_star = np.ones(10).reshape(-1, 1)
    orders = [5, 10, 15, 30]
    alphas = [.99, .999, .9999, .99999]

    for ords in orders:
        wsnr_vals = []
        for alp in alphas:
            w = construct_RLS_weights(x, y, ords, alpha=alp)
            wsnr = compute_wsnr(w, w_star)
            wsnr_vals.append(wsnr)
        plt.plot(alphas, wsnr_vals, label=f'order = {ords}')
        plt.xlabel("Alpha values (forgetting factor)")
        plt.ylabel("WSNR")
        plt.legend()
        plt.title(f"WSNR for Differing Forgetting Factors for\nDifferent Orders for Output Noise of {noise}")

    plt.show()
    return



