#coding:utf-8
from pathlib import Path
from src.subkmeans.SubKmeans import SubKmeans
from src.utils import DataIO, DataNormalization, NormalizedMutualInformation
import numpy as np
import os
import gzip
from six.moves import urllib
import operator
from datetime import datetime

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#下载mnist数据集，仿照tensorflow的base.py中的写法。
def maybe_download(filename, path, source_url):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
    return filepath

#按32位读取，主要为读校验码、图片数量、尺寸准备的
#仿照tensorflow的mnist.py写的。
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
#仿照tensorflow中mnist.py写的
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print(magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data


#抽取标签
#仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


maybe_download('train_images', 'data/mnist', SOURCE_URL+TRAIN_IMAGES)
maybe_download('train_labels', 'data/mnist', SOURCE_URL+TRAIN_LABELS)
maybe_download('test_images', 'data/mnist', SOURCE_URL+TEST_IMAGES)
maybe_download('test_labels', 'data/mnist', SOURCE_URL+TEST_LABELS)

def main_subspace_kmeans():
    """ Subspace K-means
    """
   # dataset = Path(f"./datasets/{ds}.dat")
   # dataname = dataset.stem
    print("step 1: load data...")
    #data = extract_images('data/mnist/train_images', True, True)
    #label = extract_labels('data/mnist/train_labels')
    data = extract_images('data/mnist/test_images', True, True)
    data = data[0:500,]
    
    label = extract_labels('data/mnist/test_labels')
    label = label[0:500,]
    print(type(label))
    print(type(data))
    
   # label = extract_labels('data/mnist/test_labels')

    ## step 2: training...
    ## step 3: testing
    print("step 2: testing...")
    matchCount = 0

    nrOfClusters=10
    handler = SubKmeans()
    result = handler.runWithReplicates(data, nrOfClusters, 10)
    label_ = result.labels
    
    from munkres import Munkres 
    def maplabels(L1, L2):
        Label1 = np.unique(L1)
        Label2 = np.unique(L2) 
        nClass1 = len(Label1) 
        nClass2 = len(Label2) 
        nClass = np.maximum(nClass1, nClass2) 
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
        print("map rule:\n", index)
        temp = np.array(L2)
        newL2 = np.zeros(temp.shape, dtype=int) 
        for i in range(nClass2): 
            for j in range(len(temp)): 
                if temp[j] == index[i, 0]: 
                    newL2[j] = index[i, 1]
        return newL2
   
    label_ = maplabels(label, label_)
    print("accuracy")
    from sklearn.metrics import accuracy_score
    print(accuracy_score(label, label_))
    print("nmi")
    from sklearn.metrics import normalized_mutual_info_score
    print(normalized_mutual_info_score(label, label_))
    print("Rand")
    from sklearn.metrics import adjusted_rand_score
    print(adjusted_rand_score(label, label_))
    





if __name__ == "__main__":
    main_subspace_kmeans()
