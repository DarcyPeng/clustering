from math import log, sqrt
import numpy as np
from itertools import groupby

"""
    Evaluation of the clustering performance of models based on the normalized mutual information
    Same as in scikit learn
    See https://github.com/scikit-learn/scikit-learn/blob/412996f/sklearn/metrics/cluster/supervised.py#L712
"""

def forLabelSeq(trueLabels, predictedLabels):
    assert len(trueLabels) == len(predictedLabels)
    X = dict((k, list(zip(*g))[1]) for k, g in groupby(sorted(zip(trueLabels, range(len(trueLabels))), key=lambda x: x[0]), key=lambda x: x[0]))
    Y = dict((k, list(zip(*g))[1]) for k, g in groupby(sorted(zip(predictedLabels, range(len(predictedLabels))), key=lambda x: x[0]), key=lambda x: x[0]))
    if len(X) == len(Y) and (len(X) == 0 or len(X) == 1):
        return 1
    else:
        return forClusterMap(X, Y, len(trueLabels))

def selfInformation(p: float):
    return 0 if p <= 0 else -p * log(p)

def labelEntropy(clustering: dict, nrOfDps: int):
    return sum([selfInformation(len(xi) / nrOfDps) for xi in clustering.values()])

def mutualInformation(realClusters: dict, foundClusters: dict, nrOfDps: int):
    summ = 0.0
    for xIds in realClusters.values():
        for yIds in foundClusters.values():
            pij = len(set(xIds).intersection(yIds)) / nrOfDps
            pi = len(xIds) / nrOfDps
            pj = len(yIds) / nrOfDps
            partMi = 0 if pij <= 0 else pij * log(pij / (pi * pj))
            summ += partMi

    return summ

def forClusterMap(realClusters: dict, foundClusters: dict, nrOfDps: int):
    hx = labelEntropy(realClusters, nrOfDps)
    hy = labelEntropy(foundClusters, nrOfDps)
    miScore = mutualInformation(realClusters, foundClusters, nrOfDps)
    return miScore / max(sqrt(hx * hy), 1e-10)

def forClusterSet(realClusters: list, foundClusters: list, nrOfDps: int):
    rc = dict(enumerate(realClusters))
    fc = dict(enumerate(foundClusters))
    return forClusterMap(rc, fc, nrOfDps)
    