import numpy as np
from src.utils import DataSetStat


def centerData(data: np.ndarray):
    mean = DataSetStat.dataMean(data)
    return data - mean, mean


def handleZeroVariance(varVec: np.ndarray):
    varVec[varVec == 0] = 1e-20
    return varVec

def standardizeData(data: np.ndarray):
    zeroMeanData, _ = centerData(data)
    singleDimStd = handleZeroVariance(np.sqrt((zeroMeanData ** 2).sum(axis=0) / data.shape[0]))
    return zeroMeanData / singleDimStd
    