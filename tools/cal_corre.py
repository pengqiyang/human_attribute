import os
import sys
from PIL import Image
import pickle
import torch
import numpy  as np
'''
from sklearn import preprocessing
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
'''
import pdb
'''
sim = []
labels = open('/media/data1/pengqy/dataset/pa100k/train.txt').readlines()
for line in labels:
    items = line.split()
    img_name = items.pop(0)
    cur_label = tuple([int(v) for v in items])
    sim.append(cur_label)
pdb.set_trace()
sim = np.array(sim)
sim_l2 = preprocessing.normalize(sim, norm='l2', axis=0)
sim_l2_t = sim_l2.transpose(1,0)
np.dot(sim_l2_t, sim_l2)
'''

dataset_info = pickle.load(open('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/dataset.pkl', 'rb+'))
img_id = dataset_info.image_name
img_idx = dataset_info.partition['train']
attr_label = dataset_info.label
if isinstance(img_idx, list):
    img_idx = img_idx[0]  # default partition 0
img_num = img_idx.shape[0]
img_id = [img_id[i] for i in img_idx]
label = attr_label[img_idx]
attr_name = dataset_info.attr_name

for i in range(35):
    print(str(i)+' '+attr_name[i])
#calculate the IOU
def func_1():
    dataset_info = pickle.load(open('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PA100k/dataset.pkl', 'rb+'))
    img_id = dataset_info.image_name
    img_idx = dataset_info.partition['train']
    attr_label = dataset_info.label
    if isinstance(img_idx, list):
        img_idx = img_idx[0]  # default partition 0
    img_num = img_idx.shape[0]
    img_id = [img_id[i] for i in img_idx]
    label = attr_label[img_idx]
    attr_name = dataset_info.attr_name
    label = np.array(label)
    cishu  = [0] * 26

    co_currence = [[0 for col in range(26)] for row in range(26)]
    corre=[[0 for col in range(26)] for row in range(26)]
    #for i in range(len(img_idx[0])):
    #    label.append(attr_label[i])

    #pdb.set_trace()
    label = np.array(label)
    for i in range(26):
        for j in range(26):
            co_currence[i][j] = np.sum(np.logical_and(label[:,i], label[:,j])) 
            
    for i in range(26):
        cishu[i]  = np.sum(label[:,i])
    for i  in range(26):
        for j in range(26):
            #pdb.set_trace()
            corre[i][j] = co_currence[i][j]/(np.sqrt(cishu[i]) * np.sqrt(cishu[j]))
            corre[i][j] = round(corre[i][j], 2)
    pdb.set_trace()      
    x_index, y_index = np.where(np.array(corre)>0.7)
    #shuchu = np.where(np.array(corre)>0.6 , 1,0)
    x_index_inverse, y_index_inverse = np.where(np.array(corre)==0.0)
    #for i in range(35):
    #    shuchu[i][i]=0

    #np.save('corre.npy', shuchu)
    #np.save('uncorre.npy', shuchu2)
               
        
        
    for i in range(len(x_index)):
        print(attr_name[x_index[i]] +' '+str(x_index[i])+" " +(attr_name[y_index[i]]) +" "+str(y_index[i])+ ": " + str(corre[x_index[i]][y_index[i]]))        

    print("fu xiangguanng")
    for i in range(len(x_index_inverse)):
        print(attr_name[x_index_inverse[i]] +' '+str(x_index_inverse[i])+" " +(attr_name[y_index_inverse[i]]) +" "+str(y_index_inverse[i])+ ": " + str(corre[x_index_inverse[i]][y_index_inverse[i]]))

    #np.save('corre.npy', shuchu)
def softmax(x):
    totalSum = np.sum(np.exp(x), axis = 1)
    return np.exp(x)/totalSum
    
def func_2():
    dataset_info = pickle.load(open('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PA100k/dataset.pkl', 'rb+'))
    img_id = dataset_info.image_name
    img_idx = dataset_info.partition['train']
    attr_label = dataset_info.label
    if isinstance(img_idx, list):
        img_idx = img_idx[0]  # default partition 0
    img_num = img_idx.shape[0]
    img_id = [img_id[i] for i in img_idx]
    label = attr_label[img_idx]
    attr_name = dataset_info.attr_name
    label = np.array(label)
    cishu  = [0] * 26

    co_currence = [[0 for col in range(26)] for row in range(26)]
    corre=[[0 for col in range(26)] for row in range(26)]
    #for i in range(len(img_idx[0])):
    #    label.append(attr_label[i])

    #pdb.set_trace()
    label = np.array(label)
    for i in range(26):
        for j in range(26):
            co_currence[i][j] = np.sum(np.logical_and(label[:,i], label[:,j])) 
            
    for i in range(26):
        cishu[i]  = np.sum(label[:,i])
    for i  in range(26):
        for j in range(26):
            #pdb.set_trace()
            corre[i][j] = co_currence[i][j]/cishu[i]
            corre[i][j] = round(corre[i][j], 2)
    #pdb.set_trace()    

    for i in range(26):
        for j in range(26):
            if (corre[i][j]==1):
                corre[i][j] = 1.5
            elif corre[i][j]==0.0:
                corre[i][j] = -0.35
                
            else:
                corre[i][j] = 0
    pdb.set_trace()
    #shuchu = np.where(np.array(corre)>0.7 , 2,0)

    #corre = softmax(corre)
    np.save('fc_init_pa100k.npy', corre)    
        
    #for i in range(len(x_index)):
    #   print(attr_name[x_index[i]] + " " +(attr_name[y_index[i]]) + ": " + str(corre[x_index[i]][y_index[i]]))        

    #np.save('corre.npy', shuchu)
def correlation_matrix():
    dataset_info = pickle.load(open('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/dataset.pkl', 'rb+'))
    img_id = dataset_info.image_name
    img_idx = dataset_info.partition['train']
    attr_label = dataset_info.label
    if isinstance(img_idx, list):
        img_idx = img_idx[0]  # default partition 0
    img_num = img_idx.shape[0]
    img_id = [img_id[i] for i in img_idx]
    label = attr_label[img_idx]
    attr_name = dataset_info.attr_name
    pdb.set_trace()
    label = np.array(label)
    label = label.T
    corre = np.corrcoef(label)
    for  i in range(35):
        print('name:   '+attr_name[i])
        print('负相关:')
        cc= np.argsort(corre[i])
        for j in cc[:8]:
            print(attr_name[j]+ ' ' + str(corre[i][j]))
        print('正相关:')
        for j in cc[27:]:
            print(attr_name[j]+ ' ' + str(corre[i][j]))
    #np.save('corre_peta_xiangguan.npy', corre)    
    #corre=[[0 for col in range(35)] for row in range(35)]

def generate_origin_gongxian():
    
    dataset_info = pickle.load(open('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/dataset.pkl', 'rb+'))
    img_id = dataset_info.image_name
    img_idx = dataset_info.partition['train']
    attr_label = dataset_info.label
    if isinstance(img_idx, list):
        img_idx = img_idx[0]  # default partition 0
    img_num = img_idx.shape[0]
    img_id = [img_id[i] for i in img_idx]
    label = attr_label[img_idx]
    #pdb.set_trace()
    attr_name = dataset_info.attr_name
    
    label = np.array(label)
    cishu  = [0] * 35
    
    co_currence = [[0 for col in range(35)] for row in range(35)]
    corre=[[0 for col in range(35)] for row in range(35)]
    #for i in range(len(img_idx[0])):
    #    label.append(attr_label[i])

    #pdb.set_trace()
    label = np.array(label)
    for i in range(35):
        for j in range(35):
            co_currence[i][j] = np.sum(np.logical_and(label[:,i], label[:,j])) 
            
    for i in range(35):
        cishu[i]  = np.sum(label[:,i])
    for i in range(35):
        for j in range(35):
            #pdb.set_trace()
            corre[i][j] = co_currence[i][j]/(np.sqrt(cishu[i]) * np.sqrt(cishu[j]))
            corre[i][j] = round(corre[i][j], 2)
    


    #pdb.set_trace()
    '''
    corre  =  corre/np.sum(corre, axis=1)
    for i in range(35):
        for j in range(35):
            #pdb.set_trace()
            #corre[i][j] = co_currence[i][j]/(cishu[i])
            corre[i][j] = round(corre[i][j], 2)    
    '''
    
    mask= [[0 for col in range(35)] for row in range(35)]
    mask_corre = [[0 for col in range(35)] for row in range(35)]
    non_mask = [[0 for col in range(35)] for row in range(35)]
    for i in range(35):
        for j in range(35):
            if (corre[i][j]>=0.7 and i!=j):
                #corre[i][j]= 1
                mask_corre[i][j]= 1
            if corre[i][j]==0.0:
                #corre[i][j] = -1
                
                mask[i][j] = 1
            if  0.1<corre[i][j] and corre[i][j]<0.5:
                non_mask[i][j]=1
    
    pdb.set_trace()

    
    
    #pdb.set_trace()
    #corre = np.load('/home/pengqy/paper/resnet50_corre_2/PETA/PETA/img_model/corre.npy')
    '''
    for  i in range(35):
          print('\n')
          print('name:   '+attr_name[i]+ ' '+ str(i))
          print('负相关:')
          cc= np.argsort(corre[i])
          for j in cc[:8]:
               print(attr_name[j]+' '+ str(corre[i][j])+' '+str(j))
          print('正相关:')
          for j in cc[27:]:
               print(attr_name[j]+ ' ' + str(corre[i][j])+' '+str(j))
    '''
    #pdb.set_trace()
    #np.save('gongxian_norm.npy', corre)
    np.save('nonxiangguan.npy', non_mask)
    #np.save('xiangguan.npy', mask_corre)
        
    #np.save('uncorre.npy', shuchu2)    
if __name__ == '__main__':
    #func_2()
    generate_origin_gongxian()
    #correlation_matrix()
