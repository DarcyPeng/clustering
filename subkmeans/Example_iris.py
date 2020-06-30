import numpy as np
from pathlib import Path
from src.subkmeans.SubKmeans import SubKmeans
from src.utils import DataIO, DataNormalization, NormalizedMutualInformation
from src.K_means import k_means
import pandas as pd
# for ecoli data
path =  r'C:\\Users\\22186\\Desktop\\Teamproject\\dataset\\iris.data'
df = pd.read_table(path,encoding='utf-8',sep=',',header=None)

def main_subspace_kmeans(ds, separator=";"):
    """ Subspace K-means
    """
    dataset = Path(f"./datasets/{ds}.dat")
    dataname = dataset.stem
    from sklearn.preprocessing import LabelEncoder
    df[df.columns[-1]] = LabelEncoder().fit_transform(df[df.columns[-1]])
    data = df[df.columns[0:-1]]
    label = df[df.columns[-1]]
    from sklearn.preprocessing import MinMaxScaler
    data = MinMaxScaler().fit_transform(data)
    #d, groundTruth = DataIO.loadCsvWithIntLabelsAsSeq(dataset, separator=separator)
    #data = DataNormalization.standardizeData(d)
    nrOfClusters=len(label.unique())
   # nrOfClusters = np.unique(groundTruth).shape[0]
    #print("类别：",nrOfClusters)
    handler = SubKmeans()
    result = handler.runWithReplicates(data, nrOfClusters, 10)
    label_ = result.labels
    from munkres import Munkres 
    def maplabels(L1, L2):
        Label1 = np.unique(L1)
        #print("Label1:",Label1)
        Label2 = np.unique(L2) 
        #print("Label2:",Label2)
        nClass1 = len(Label1) 
        nClass2 = len(Label2) 
        #print("nClass1:",nClass1)
        #print("nClass2:",nClass2)
        nClass = np.maximum(nClass1, nClass2) 
        #print("nClass:",nClass)
        G = np.zeros((nClass, nClass))

        for i in range(nClass1): 
            ind_cla1 = L1 == Label1[i] 
            ind_cla1 = ind_cla1.astype(float) 
            for j in range(nClass2): 
                ind_cla2 = L2 == Label2[j] 
                ind_cla2 = ind_cla2.astype(float) 
                G[i, j] = np.sum(ind_cla2*ind_cla1)
        m = Munkres() 
        index = m.compute(-G.T) 
        index = np.array(index) # print(-G.T) 
        #print("map rule:\n", index)
        temp = np.array(L2)
        newL2 = np.zeros(temp.shape, dtype=int) 
        for i in range(nClass2): 
            for j in range(len(temp)): 
                if temp[j] == index[i, 0]: 
                    newL2[j] = index[i, 1]
        return newL2
   
    
    #print("result_acc:",result_acc)
    
   # print(type(groundTruth))
    #print("groundTruth",groundTruth)
    #print("label:",label)
    #print("label_",label_)
    label_ = maplabels(label, label_)
    #print("newlabel:",label)
   # print("newlabel_",label_)
    print("accuracy")
    from sklearn.metrics import accuracy_score
    print(accuracy_score(label, label_))
    print("nmi")
    from sklearn.metrics import normalized_mutual_info_score
    print(normalized_mutual_info_score(label, label_))
    print("Rand")
    from sklearn.metrics import adjusted_rand_score
    print(adjusted_rand_score(label, label_))
    '''
    nmi = NormalizedMutualInformation.forLabelSeq(label, label_)
    print(f"NMI: {nmi}")
    print(f"m: {result.m()}")
    '''
    transformedData = np.matmul(result.transformation().T, data[:, :, None]).squeeze()
    DataIO.writeClusters(Path("./result") / f"{dataname}_result.dat", transformedData, result.labels)


def main_kmeans(ds, separator=";"):
    """ K-means
    """
    dataset = Path(f"./datasets/{ds}.dat")

    d, groundTruth = DataIO.loadCsvWithIntLabelsAsSeq(dataset, separator=separator)
    data = DataNormalization.standardizeData(d)

    nrOfClusters = np.unique(groundTruth).shape[0]

    all_nmi = 0
    for _ in range(10):
        result_labels = k_means(data, nrOfClusters)
        nmi = NormalizedMutualInformation.forLabelSeq(groundTruth, result_labels)
        all_nmi += nmi
    
    print(f"NMI: {all_nmi / 10}")


if __name__ == "__main__":
    args = ["iris", ","]
    #main_kmeans(*args)
    main_subspace_kmeans(*args)
