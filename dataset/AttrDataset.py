from __future__ import absolute_import
import os
import pickle
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import pdb
from tools.function import get_pkl_rootpath
import torchvision.transforms as T


from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class Random_Semantic_Erasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, name =' ', probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.name = name
    #给定激活点，以激活点为中心，生成矩形      
    '''
    def __call__(self, img):
        #pdb.set_trace()
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in loc[self.name]:
            x, y = attempt[0], attempt[1]
            h = random.choice(select_h)
        
            if 0<=x-h and x+h <= img.size()[1]:
                img[0, x-h:x+h, y-h:y+h] = self.mean[0]
                img[1, x-h:x+h, y-h:y+h] = self.mean[1]
                img[2, x-h:x+h, y-h:y+h] = self.mean[2]
                
               
        return img
    '''
    
    #根据CAM生成mask
    def __call__(self, img):
        #pdb.set_trace()
        if random.uniform(0, 1) > self.probability:
            return img
        #pdb.set_trace()
        if (os.path.isfile('RAP_mask/'+self.name.split('.')[0]+'.npy')==False):
            return img
        loc = np.load('RAP_mask/'+self.name.split('.')[0]+'.npy', allow_pickle=True).item()
        if len(loc['cam'])==0:
        
            return img
        mask = 0
        total = 0
        for attempt in loc['cam']:
            if random.uniform(0, 1) > self.probability:
                attempt = np.where(attempt>=0.95, 1,0)
                mask = mask + attempt
                total = total + 1
       
        if total == 0:
            return img
        mask = (1- mask/total)
        #cv2.imwrite('0.jpg', np.uint8(255*mask))
        #pdb.set_trace()
        img = torch.from_numpy(mask).float() * img       
        return img
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        
        

    
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img



class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None, Type='train'):

        assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))
    
        self.Type = Type
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root
        #self.root_path = 'data/RAP_HEAD'
        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        #pdb.set_trace()
        self.label = attr_label[self.img_idx]
        self.attr_name = dataset_info.attr_name

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)
        
        '''
        #pdb.set_trace()
        if self.Type == 'train' or self.Type == 'trainval':
            
            img = T.Resize((256, 192))(img)
            img = T.ToTensor()(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            #pdb.set_trace()
            img = Random_Semantic_Erasing(imgname)(img)
        
        elif self.Type == 'val' or self.Type == 'test':
         
            img = T.Resize((256, 192))(img)
            img = T.ToTensor()(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)           
        '''
        if self.transform is not None:
            img = self.transform(img)
        
        
        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    #normalize = T.Normalize(mean=[0.6425], std=[0.2979])
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        # T.Pad(10),
        # T.RandomCrop((height, width)),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
