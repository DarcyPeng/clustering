import numpy as np
from functools import reduce
from itertools import groupby

from src.utils import DataSetStat, MatrixUtils, NormalizedMutualInformation
from src.subkmeans.AbstractClustering import (
    AbstractClustering, SubKmeansCluster, InitClusterCentroids, 
    Config, Costs, Results
)


class SubKmeans(AbstractClustering):
    """
    A basic and faithful implementation of the proposed SubKmeans algorithm.
    For brevity and clarity:
    it does not contain simple optimization strategies like cashing the rotated data points.
    it does not contain a strategy for handling empty clusters
    it assumes that the clustered space contains at least one feature (m>=1)
    it implements each step as a self contained function and sometimes with the side-effect that some values are calculated multiple times
    """
    def __init__(self):
        self.MAX_NR_OF_ITERATIONS = 200
        self.MAX_NMI_DIFFERENCE = 0.0001
        self.ROUNDS_WITHOUT_PRUNING_M = 2
        super(SubKmeans, self).__init__()

    def runWithReplicates(self, data: np.ndarray, k: int, replicates: int):
        """ Runs SubKmeans multiple times with a for sample datasets sufficient default configuration and returns the result with minimal costs
        """
        #print("runWithReplicate k:",k)
        results = []
        for _ in range(replicates):
            init = self.initConfig(data, k, InitClusterCentroids.useKmpp)
            results.append(self.run(init))

        return min(results, key=lambda x: x.finalCosts.total)

    def run(self, initConfig: Config):
        """ This function executes the main algorithm for the given config
        """
        currentConfig = initConfig
        counter = 0

        converged = False
        lastLabels = []
        nrOfDps = initConfig.data.shape[0]

        while True:
            counter += 1

            # Assigns data points to the nearest cluster center
            currentConfig = self.dpAssignmentStep(currentConfig)
            # Update cluster centers
            currentConfig = self.updateClusterCenters(currentConfig)
            # Update transformation matrix and m
            currentConfig = self.updateTransformation(currentConfig, counter >= self.ROUNDS_WITHOUT_PRUNING_M)

            # We determine convergence with the help of NMI
            currentLabels = [cl.dataPoints for cl in currentConfig.clusters]
            converged = NormalizedMutualInformation.forClusterSet(currentLabels, lastLabels, nrOfDps) > 1 - self.MAX_NMI_DIFFERENCE
            lastLabels = currentLabels

            if not converged and counter < self.MAX_NR_OF_ITERATIONS or (counter <= self.ROUNDS_WITHOUT_PRUNING_M):
                continue
            else:
                break

        costs = self.vanillaCosts(currentConfig)
        print(f"it took {counter} rounds and {costs}")
        return Results(self.findClusterAssignment(currentConfig), costs, currentConfig)

    def initConfig(self, data: np.ndarray, k: int, clusterSampler, initM=lambda x: int(round(x / 2))):
        """
        Initializes a configuration based on the given parameters

        @param initM excepts a function which determines the initial cluster space dimensionality based on the given dataset dimensionality.
                     The default value is d/2
        """
        # 获取原始空间的特征数量
        nrOfDims = data.shape[1]
        #print("nrOfDims:",nrOfDims)
        m = min(max(2, initM(nrOfDims)), nrOfDims)
        #print("11111")
        #print("initialization m:",m)
        initCandidates = []
        oldRand = np.random.randint(0x80000000, size=(10,))
        for i in range(10):
            np.random.seed(oldRand[i])
           # print("initCR k:",k)
            initCR = self.initRotationAndClusters(data, k, m, clusterSampler)
            randRotation, initClusters = initCR.rotationMatrix, initCR.clusters
            clusters = [SubKmeansCluster(*DataSetStat.dataMeanAndScatter(np.array(x[0])), x[1]) 
                        for x in initClusters]
            # find mapping
            allMean, scatterMatrix = DataSetStat.dataMeanAndScatter(data)
            init = Config(k, m, clusters, allMean, scatterMatrix, randRotation, np.ones((nrOfDims, )), data)
            initCandidates.append(self.updateTransformation(init, pruneM=False))
        
        return min(initCandidates, key=lambda x: self.vanillaCosts(x).total)
        
    def determineMFromEigenvalues(self, eigenvals: np.ndarray):
        """
        Determines the parameter m based on the eigenvalues.
        This function assumes that components with small negative eigenvalues should point towards the noise space
        """
        return max(np.count_nonzero(eigenvals / eigenvals[0] > 1e-8), 1)

    def updateTransformation(self, config: Config, pruneM: bool):
        """
        Updates the transformation V and the dimensionality of the clustered space m
        The later only happens if pruneM is true
        """
        setupA = sum([x.scatterMatrix for x in config.clusters]) - config.scatterMatrixAllData
        eigenvalues, eigenvectors = MatrixUtils.sortedEigSym(setupA, ascending=True)
        newM = self.determineMFromEigenvalues(eigenvalues) if pruneM else config.m

        return config.copy(transformation=eigenvectors, m=newM, eigenvalues=eigenvalues)
    
    def vanillaCosts(self, config: Config):
        dims = config.nrOfDims
        data = config.data
        m = config.m
        Pc = self.calcPc(dims, m)   # [dims, m]
        Pn = self.calcPn(dims, m)   # [dims, m]

        clusterSpaceMappingT = np.matmul(config.transformation, Pc).T   # [m, dims]
        noiseSpaceMappingT = np.matmul(config.transformation, Pn).T     # [m, dims]

        # We might want to cache the results here (not done for brevity)
        clusterMeansInClSpace = [np.matmul(clusterSpaceMappingT, x.fullDimMean) for x in config.clusters]

        noiseClusterMappedMean = np.matmul(noiseSpaceMappingT, config.fullDimNoiseMean)     # [m,]

        noiseCosts = ((np.matmul(noiseSpaceMappingT, data.T).T - noiseClusterMappedMean) ** 2).sum()
        clusterCosts = 0.0
        
        def compare(bestCurrent, clSpaceClMean):
            res = ((np.matmul(clusterSpaceMappingT, dpr) - clSpaceClMean) ** 2).sum()
            if bestCurrent < res:
                return bestCurrent
            else:
                return res

        for dpr in data:
            clusterCostsPart = reduce(compare, clusterMeansInClSpace, np.Inf)
            clusterCosts += clusterCostsPart

        return Costs(noiseCosts, clusterCosts)

    def dpAssignmentStep(self, config: Config):
        """ Assigns data points to the nearest cluster center
        """
        assignment = self.findClusterAssignment(config)
        # {0: (8, 12, 16, 19), ...}
        groupedLabels = dict((k, list(zip(*g))[1]) for k, g in groupby(sorted(zip(assignment, range(len(assignment))), key=lambda x: x[0]), key=lambda x: x[0]))
        nextClusters = [cl.withNewDataPoints(groupedLabels[idx]) for idx, cl in enumerate(config.clusters) if idx in groupedLabels]
        if config.k != len(config.clusters):
            print(f"Warning we lost some clusters: k={config.k} but currently we have {len(config.clusters)}")
        
        return config.withNewClusters(nextClusters)

    def findClusterAssignment(self, config: Config):
        """ Belongs to dpAssignmentStep 
        """
        data = config.data
        m = config.m
        Pc = self.calcPc(config.nrOfDims, m)
        clusterSpaceMappingT = np.matmul(Pc.T, config.transformation.T)     # [m, dims]
        
        # [[m,]]
        mappedClusters = [np.matmul(clusterSpaceMappingT, x.fullDimMean) for x in config.clusters]
        mappedClusters = np.array(mappedClusters)   # [k, m]
        
        newAssignment = [((np.matmul(clusterSpaceMappingT, dpr) - mappedClusters) ** 2).sum(axis=1).argmin() 
                         for dpr in data]

        return newAssignment

    def updateClusterCenters(self, config: Config):
        data = config.data
        newMeanClusters = [cl.withNewMeanAndScatter(*DataSetStat.dataMeanAndScatter(data[cl.dataPoints])) for cl in config.clusters]
        return config.withNewClusters(newMeanClusters)

    
