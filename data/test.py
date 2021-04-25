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
#pdb.set_trace()
#dirs = os.listdir('output_monodepth/')

dirs = os.listdir('PA100k_DPT/')
count = 0
for name in dirs:
    if count % 1000 == 0:
        #pdb.set_trace()
        #img = cv2.imread('output_monodepth/'+str(name), 0)
        img_1 = Image.open('PA100k_DPT/' +  str(name))
        imgs.append(transform_1(img_1))
        #pdb.set_trace()
        #cv2.imwrite('PA100k_DPT/'+name.split('.')[0]+'.jpg', img)
        #imgs.append(transform_1(img_1))
    count = count+1
pdb.set_trace()
torch_imgs = torch.cat(imgs, dim=0)
print(torch.mean(torch_imgs))
print(torch.std(torch_imgs))
#img_2 = cv2.imread('PA100k/data/100000.jpg')
#print(name)
#cv2.imwrite('PA100k_DPT/'+str(name), img_1)
#img = Image.open('1.jpg')
#ig = T.ToTensor()(img) 
#print('done')


#print(img_1)
#print(img_2)
#i_1 = transform_1(img)
#i_2 = transform_1(img_2)
#print(i_1)
#print(i_2)
#print(i_1)
#pdb.set_trace()

