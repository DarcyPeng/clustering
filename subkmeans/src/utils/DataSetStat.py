import numpy as np

def determineScatter(centered: "[N, ndims]"):
    # [N, ndims, 1]
    centered = centered[:, :, None]
    return np.sum(np.matmul(centered, centered.transpose(0, 2, 1)), axis=0)


def determineCov(centered: "[N, ndims]", adjustCov: bool):
    scatter = determineScatter(centered)
    if adjustCov:
        return scatter / (centered.shape[0] - 1)
    else:
        return scatter / centered.shape[0]


def dataMean(data: np.ndarray):
    return np.mean(data, axis=0)


def dataMeanAndPreFeatureVariance(data: np.ndarray):
    mean = dataMean(data)
    variance = np.sum((data - mean) ** 2, axis=0)
    return mean, variance


def dataMeanAndCov(data: np.ndarray, adjustCov: bool):
    mean = dataMean(data)
    meaned = data - mean
    cov = determineCov(meaned, adjustCov)
    return mean, cov


def dataMeanAndScatter(data: np.ndarray):
    mean = dataMean(data)
    meaned = data - mean
    scatter = determineScatter(meaned)
    return mean, scatter


def dataScatter(data: np.ndarray, assumeCentered: bool):
    centered = data if assumeCentered else data - dataMean(data)
    return determineScatter(centered)


def dataCov(data: np.ndarray, assumeCentered: bool, adjustCov: bool):
    if assumeCentered:
        return determineCov(data, adjustCov)
    else:
        return dataMeanAndCov(data, adjustCov)[1]
