import numpy as np
from copy import deepcopy
from collections import namedtuple
from functools import reduce


class AbstractClustering(object):
    """
    The trait and classes in this file provides utility functionality, 
    which is not directly needed to understand the SubKmeans algorithm
    """
    def __init__(self):
        self.InitClustersAndRotation = namedtuple("InitClustersAndRotation", ["rotationMatrix", "clusters"])

    def calcPc(self, dims: int, m: int) -> np.ndarray:
        if m > 0:
            return np.vstack((np.eye(m), np.zeros(shape=(dims - m, m))))
        else:
            return np.zeros((dims, 0))

    def calcPn(self, dims: int, m: int) -> np.ndarray:
        if m < dims:
            return np.vstack((np.zeros((m, dims - m)), np.eye(dims - m)))
        else:
            return np.zeros((m, 0))

    def initRotationAndClusters(self, data: np.ndarray, k: int, m: int, clusterSampler, rand=np.random.uniform):
        
        print("data:",data)
        nrOfDims = data.shape[1]
        print("nrofdims:",nrOfDims)
        randRotation = np.linalg.qr(rand(size=(nrOfDims, nrOfDims)))[0]
        mapping = np.matmul(self.calcPc(nrOfDims, m).T, randRotation.T)
        print("mapping:",mapping)
        print("data:",data)
        rotatedData = np.matmul(mapping, data.T).T  # n x m
        
        clusterCenters = clusterSampler(rotatedData, k)

        # dps: data points
        dps2Clusters = {idx: ([], []) for idx in range(len(clusterCenters))}
        for dpIdx in range(len(rotatedData)):
            dpr = rotatedData[dpIdx]
            _, clusterId = min(zip(clusterCenters, range(len(clusterCenters))), key=lambda mcd: np.sum((dpr - mcd[0])**2))
            clusterDpList, idList = dps2Clusters[clusterId]
            clusterDpList.append(data[dpIdx])
            idList.append(dpIdx)

        clusterValues = list(dps2Clusters.values())
        if any(map(lambda value: not value[1], clusterValues)):
            return self.initRotationAndClusters(data, k, m, clusterSampler)
        else:
            return self.InitClustersAndRotation(randRotation, clusterValues)
                
    
class SubKmeansCluster(object):
    def __init__(self, fullDimMean: np.ndarray, scatterMatrix: np.ndarray, dataPoints: "list of indices"):
        self.fullDimMean = fullDimMean
        self.scatterMatrix = scatterMatrix
        self.dataPoints = list(dataPoints)
        self.nrOfDps = len(dataPoints)
        
    def copy(self, **kwargs):
        inputs = {"fullDimMean": self.fullDimMean,
                  "scatterMatrix": self.scatterMatrix,
                  "dataPoints": self.dataPoints}
        inputs.update(kwargs)
        return SubKmeansCluster(**inputs)

    def withNewDataPoints(self, newDps: list):
        """ Update in `dpAssignmentStep`
        """
        return self.copy(dataPoints=newDps)

    def withNewMeanAndScatter(self, mean: np.ndarray, scatter: np.ndarray):
        """ Update in `updateClusterCenters`
        """
        return self.copy(fullDimMean=mean, scatterMatrix=scatter)


class Config(object):
    def __init__(self, k: int, # Nr of clusters
                       m: int, # nr of clustering dimensions
                       clusters: list,
                       fullDimNoiseMean: np.ndarray,
                       scatterMatrixAllData: np.ndarray,
                       transformation: np.ndarray, # each column represents one transformation vector for one dimension
                       eigenvalues: np.ndarray,
                       data: list):
        self.k = k
        self.m = m
        self.clusters = clusters
        self.fullDimNoiseMean = fullDimNoiseMean
        self.scatterMatrixAllData = scatterMatrixAllData
        self.transformation = transformation
        self.eigenvalues = eigenvalues
        self.data = data

        self.nrOfDims = data.shape[1]

    def copy(self, **kwargs):
        inputs = {"k": self.k,
                  "m": self.m,
                  "clusters": self.clusters,
                  "fullDimNoiseMean": self.fullDimNoiseMean,
                  "scatterMatrixAllData": self.scatterMatrixAllData,
                  "transformation": self.transformation,
                  "eigenvalues": self.eigenvalues,
                  "data": self.data}
        inputs.update(kwargs)
        return Config(**inputs)

    def withNewClusters(self, newClusters):
        """ Update in `dpAssignmentStep`
        """
        return self.copy(clusters=newClusters)


class Costs(object):
    def __init__(self, noiseData: float, clusterData: float):
        self.noiseData = noiseData
        self.clusterData = clusterData
        self.total = noiseData + clusterData

    def __str__(self):
        return f"Costs: - noiseData: {self.noiseData}  clusterData: {self.clusterData}  total: {self.total}"

    __repr__ = __str__


class Results(object):
    def __init__(self, labels: list, finalCosts: Costs, config: Config):
        self.labels = labels
        self.finalCosts = finalCosts
        self.config = config

    def m(self):
        return self.config.m

    def transformation(self):
        return self.config.transformation


class InitClusterCentroids(object):
    @classmethod
    def useKmpp(cls, data: np.ndarray, k: int, maxNrOfDpsToConsider=5000, rand=np.random):
        #print("useKmpp k:",k)

        nrOfPoints = data.shape[0]
        sampleData = data[rand.randint(0, nrOfPoints, maxNrOfDpsToConsider)] if nrOfPoints > maxNrOfDpsToConsider else data

        initIdx = rand.randint(0, sampleData.shape[0])

        dataPoints = sampleData
        initCenter = [dataPoints[initIdx]]
        remainingData = list(range(dataPoints.shape[0]))
        remainingData.remove(initIdx)
        #print("useKmpp k:",k-1)
        return cls.kppChooseNextCenter(initCenter, dataPoints, remainingData, k - 1)

    @classmethod
    def kppChooseNextCenter(cls, centers: list, data: np.ndarray, remainingData: list, stillToChoose: int, rand=np.random):
        # TODO: Not the fastest implementation...
        
        #print("stillchoose:",stillToChoose)
        
        if stillToChoose == 0:
            return centers
        
        squaredDistances = np.array([reduce(min, [((c - data[dpIdx]) ** 2).sum() for c in centers], np.Inf) for dpIdx in remainingData])

        # Warning: Remove `val probAdjusted = if (prob.isNaN) 1e-10 else prob`
        probs = squaredDistances / squaredDistances.sum()

        newCenter = np.random.choice(remainingData, p=probs)
        centers.append(data[newCenter])
        remainingData.remove(newCenter)
        return cls.kppChooseNextCenter(centers, data, remainingData, stillToChoose - 1)
