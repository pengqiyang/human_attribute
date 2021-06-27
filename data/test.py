import os
import pickle
import numpy as np
import cv2
import torch.utils.data as data
from PIL import Image
#from tools.function import get_pkl_rootpath
import torchvision.transforms as T
import torch
import pdb

normalize_1 = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize_2 = T.Normalize(mean=[2.8424503515351494], std=[0.9932836506164299])
transform_1 = T.Compose([
        T.Resize((256, 192)),
        T.ToTensor(),
        #normalize_1,
])

transform_2 = T.Compose([
        T.Resize((256, 192)),
        T.ToTensor(),
        #normalize_2,
])
imgs=[]


def calculate_mean_std():
    dirs = os.listdir('/media/data1/pengqy/Celeba/dataset/train_DPT/5221/')
    count = 0
    for name in dirs:
        #if count % 100 == 0:
            
        img_1 = Image.open('/media/data1/pengqy/Celeba/dataset/train_DPT/5221/' +  str(name))
        imgs.append(transform_1(img_1).float())
            
        #count = count+1
   
    torch_imgs = torch.cat(imgs, dim=0)
    print(torch.mean(torch_imgs))
    print(torch.std(torch_imgs))
def depth_to_img():
    #f = open('test.txt', 'w')
    dirs = os.listdir('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/RAP2_DPT/')
    count = 0
    for name in dirs:
        #f.write(name+'\n')

        img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/RAP2_DPT/'+str(name), 0)
        #img = cv2.applyColorMap(img, 2)
        #img = cv2.bilateralFilter(src=img, d=0, sigmaColor=100, sigmaSpace=15)
        cv2.imwrite('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/RAP2_DPT/'+name.split('.')[0]+'.png', img)
        
    #f.close()    
if __name__ == '__main__':

    #depth_to_img()
    calculate_mean_std()
