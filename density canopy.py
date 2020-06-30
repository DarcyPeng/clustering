from sklearn.metrics.pairwise import pairwise_distances
from functools import reduce
import numpy as np
import pandas as pd

import sys

class densityCanopy():
    def __init__(self, remove_outliers=True):
        self.remove_outliers = remove_outliers
  
    def calc_meanDist(self, data, dists=None):  
        n = data.shape[0]
        if dists is None:
            dists = pairwise_distances(data, metric='euclidean')
        triang_dists = dists[np.arange(dists.shape[0])[:,None] > np.arange(dists.shape[1])].sum()
        meanDist = 2*triang_dists/(n*(n-1))
        return dists, meanDist

    def p_density(self, dp, dists_i, meanDist):
        f = np.where((dists_i - meanDist)<0, 1, 0)
        p_i = f.sum()
        return p_i

    def a_density(self, meanDist, dists_i, p_i):
        cluster_dists = dists_i[dists_i < meanDist]
        d = cluster_dists.sum()
        if p_i-1 == 0:
            return 0
        else:
            return 2*d/(p_i*(p_i-1))

    def s_distance(self, p, p_i, dists_i):
    
        dist_i_less_j = dists_i[p-p_i>0]   
        if dist_i_less_j.size > 0:        
            return dist_i_less_j.min()
        else:       
            return dists_i.max()
      
    def w_weight(self, p_i, a_i, s_i):   
        if a_i == 0:
            return 0
        else:       
            return p_i*s_i/a_i

    def removeData(self, meanDist, dists, data, ind, centroids_dists=np.array([])): 
        dists_i = dists[ind, :] 
        dist_filter = dists_i>=meanDist  
        new_dists = dists[dist_filter, :]    
        new_dists = new_dists[:, dist_filter]
        new_data = data[dist_filter, :]

        new_centroids_dists = []
        for ind in range(centroids_dists.shape[0]):
            centroid_dists = centroids_dists[ind]
            new_centroids_dists.append(centroid_dists[dist_filter])
        new_centroids_dists = np.array(new_centroids_dists)
        return new_dists, new_data, new_centroids_dists, dist_filter

    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            data = dt.values
        elif isinstance(dt, np.ndarray):
            data = dt
        elif isinstance(dt, list):
            data = np.array(dt)
        else:
            raise Exception('dt should be a DataFrame or a numpy array')
        self.centroids = {}
        centroids_dists = np.array([])
        p_centroid = np.array([])
          
        p = np.array([[]])
        dists, meanDist = self.calc_meanDist(data)
        for ind in range(data.shape[0]):
            p = np.append(p, self.p_density(ind, dists[ind,:], meanDist))

        max_p_sample_ind = p.argmax() 
        centroid = data[max_p_sample_ind, :]
    
        centroid_index = 0
        self.centroids[centroid_index] = centroid  

        centroids_dists = np.concatenate([centroids_dists, dists[max_p_sample_ind, :]],
                                     axis=0).reshape(1,-1)

        p_centroid = np.append(p_centroid, p[max_p_sample_ind]) 
        dists, data, centroids_dists, dist_filter = self.removeData(meanDist, dists, data, max_p_sample_ind, centroids_dists=centroids_dists)
        p = p[dist_filter] 

        p = np.array([])
        a = np.array([])
        s = np.array([])
        w = np.array([])

        _, meanDist = self.calc_meanDist(data, dists=dists)   
        for ind in range(data.shape[0]):
            p_i = self.p_density(ind, dists[ind,:], meanDist) 
            p = np.append(p, p_i)          
            a = np.append(a, self.a_density(meanDist, dists[ind,:], p_i))

    
        for ind in range(data.shape[0]):
            s_i = self.s_distance(p, p[ind], dists[ind,:])
            s = np.append(s, s_i)         
            w = np.append(w, self.w_weight(p[ind], a[ind], s_i))
       
        max_w_sample_ind = w.argmax()        
        centroid = data[max_w_sample_ind, :]
       
        centroid_index += 1

        self.centroids[centroid_index] = centroid
        centroids_dists = np.concatenate([centroids_dists, [dists[max_w_sample_ind, :]]],
                                     axis=0)        
        p_centroid = np.append(p_centroid, p[max_w_sample_ind])       
        dists, data, centroids_dists, dist_filter = self.removeData(meanDist, dists, data, max_w_sample_ind, centroids_dists=centroids_dists)


        p_prev = p[dist_filter]
        s_prev = s[dist_filter]
        p = p[dist_filter]
        a = a[dist_filter]

        c_remove = 0

        while data.shape[0] > 1:
            w = np.array([])
            ind = 0            
            _, meanDist = self.calc_meanDist(data, dists=dists)
            if meanDist == 0:
                break
                
            while (ind < data.shape[0]):
                p_i = self.p_density(ind, dists[ind, :], meanDist)
                a_i = self.a_density(meanDist, dists[ind, :], p_i)
                s_centroid = []
                w_i = 1

                for centroid_dists in centroids_dists:
                    s_i = centroid_dists[ind]
                    s_centroid.append(s_i)
                    w_i *= self.w_weight(p_i, a_i, s_i)

                
                if p_prev[ind] > p_i and s_prev[ind] < min(s_centroid) and self.remove_outliers:
                    c_remove += 1
                    
                    data = np.delete(data, ind, axis=0)
                    dists = np.delete(dists, ind, axis=0)
                    dists = np.delete(dists, ind, axis=1)
                    p = np.delete(p, ind)
                    a = np.delete(a, ind)
                    
                    p_prev = np.delete(p_prev, ind)
                    s_prev = np.delete(s_prev, ind)
                   
                    centroids_dists = np.delete(centroids_dists, ind, axis=1)
                    _, meanDist = self.calc_meanDist(data, dists=dists)                    
                else:
                    p_prev[ind] = p_i
                    s_prev[ind] = min(s_centroid)
                    w = np.append(w, w_i)
                    ind += 1
                
            if w.size > 0:
                max_w_sample_ind = w.argmax()
                centroid = data[max_w_sample_ind, :]
                centroids_dists = np.concatenate([centroids_dists, [dists[max_w_sample_ind, :]]], axis=0)
                p_centroid = np.append(p_centroid, p[max_w_sample_ind])
               
                centroid_index += 1
                self.centroids[centroid_index] = centroid             

                dists, data, centroids_dists, dist_filter = self.removeData(meanDist, dists, data, max_w_sample_ind,
                                                                  centroids_dists=centroids_dists)
                p_prev = p_prev[dist_filter]
                s_prev = s_prev[dist_filter]
                p = p[dist_filter]
                a = a[dist_filter]

        print('Canopy found %d centers' % (len(self.centroids)))      