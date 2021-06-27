import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
import pdb
import cv2
import os
import torch
'''
list_filt = os.listdir('../npy') 
for fi in list_filt:
    cc= np.load('../npy/'+fi)
    cc = cc.reshape(64, -1).transpose(1,0)
    #pdb.set_trace()
    #cc -= np.mean(cc, axis = 0) # 减去均值，使得以0为中心
    #cc /= (1e-6+np.std(cc, axis = 0)) # 归一化
    #estimator = PCA(n_components=64)
    #cc = estimator.fit_transform(cc)
    #dataset_info = pickle.load(open('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/dataset.pkl', 'rb+'))
    #pdb.set_trace()
    kmeans = KMeans(2, random_state=0)
    kmeans.fit(cc)
    #pdb.set_trace()
    cc = kmeans.labels_.reshape(128, 96)
    cc =  1 - (cc ^ cc[64][48])

    cv2.imwrite(fi+'.jpg', 255*cc)

'''
'''
lists = [[] for i in range(2)]
for i in range(len(list(kmeans.labels_))):
    #lists[kmeans.labels_[i]].append(dataset_info.attr_name[i])
    lists[kmeans.labels_[i]].append(i)
print(lists)
'''

'''


S = cc.dot(cc.transpose(1,0))
#y_pred = sklearn.cluster.affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200, damping=0.5, copy=True, verbose=False, return_n_iter=False)

af = AffinityPropagation().fit(S)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
print(cluster_centers_indices)
print(labels)


lists = [[] for i in range(7)]
for i in range(len(list(labels))):
    #lists[kmeans.labels_[i]].append(dataset_info.attr_name[i])
    lists[labels[i]].append(i)
print(lists)
'''


cc= np.load('../part_detector_layer4.npy', allow_pickle=True)
#pdb.set_trace()
for i in range(0, 35):
    '''
    if i == 14:
        np.save(str(i)+'.npy',np.array(cc[i])[:,:,0,0])
    else:
    '''
    print(i)
    #pdb.set_trace()
    temp = np.array(cc[i])[:,:,0]
    #pdb.set_trace()
    temp = torch.nn.functional.normalize( torch.from_numpy(temp), p=2, dim=1, eps=1e-12, out=None)
    
    kmeans = KMeans(3, random_state=0)
    kmeans.fit(temp.numpy())
    '''
    dist = np.sqrt(np.sum(np.square(temp - kmeans.cluster_centers_[0]), axis=1))
    index_0 = np.argmin(dist)
    dist = np.sqrt(np.sum(np.square(temp - kmeans.cluster_centers_[1]), axis=1))
    index_1 = np.argmin(dist)
    dist = np.sqrt(np.sum(np.square(temp - kmeans.cluster_centers_[2]), axis=1))
    index_2 = np.argmin(dist)   
    temp = np.stack([temp[index_0], temp[index_1], temp[index_2]])       
    '''
    #kmeans.cluster_centers_
    np.save('layer4/'+str(i)+'.npy', kmeans.cluster_centers_)
