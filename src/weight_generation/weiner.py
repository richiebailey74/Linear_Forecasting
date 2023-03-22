from src.weight_generation_helpers import generateCrossCorrVec, generateAutoCorrMat, generateAutoCorrMatSpeech, generateCrossCorrVecSpeech
import numpy as np


def generateWienerWeights(x, y, order, window):
    R = generateAutoCorrMat(x, order, window)
    P = generateCrossCorrVec(x, y, order, window)
    W_n = np.linalg.inv(.001 * np.eye(order) + R) @ P  # adding small values along diagonal to prevent insufficient rank

    return W_n


def generateWienerWeightsSpeech(x, y, order, window):
    R = generateAutoCorrMatSpeech(x, order, window)
    P = generateCrossCorrVecSpeech(x, y, order, window)
    W_n = np.linalg.inv(.001 * np.eye(order) + R) @ P  # adding small values along diagonal to prevent insufficient rank

    return W_n
