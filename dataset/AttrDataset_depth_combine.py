import os
import pickle
import pdb
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
from tools.function import get_pkl_rootpath
import skimage.transform
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms as T
import cv2
import random
image_h = 256
image_w = 192
height = 256
width = 256
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize_peta = T.Normalize(mean=[0.6425], std=[0.2979])
class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {args.dataset} is not exist'

        self.data_name = args.dataset
        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label
        #pdb.set_trace()
        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
       
        self.target_transform = target_transform
        self.depth_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize_peta,
        ])
        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)
        self.attr_name = dataset_info.attr_name

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        
        if self.data_name == 'PA100k':
            depthpath = os.path.join('data/PA100k_DPT', imgname.split('.')[0]+'.png')
        elif self.data_name == 'PETA':
            depthpath = os.path.join('data/PETA_DPT', imgname)
        elif self.data_name == 'RAP':
            depthpath = os.path.join('data/RAP_DPT', imgname)
        elif self.data_name == 'RAP2':
            depthpath = os.path.join('data/RAP2_DPT', imgname)       
        '''  
        img = cv2.imread(imgpath)
        depth = cv2.imread(depthpath, 0)
        
        sample = {'image': img, 'depth': depth}
        '''
        img = Image.open(imgpath)
        depth = Image.open(depthpath)
        #pdb.set_trace()
        if self.transform is not None:
            ''''
            output = self.transform(sample)
            img = output['image']
            depth = output['depth']
            '''
            img = self.transform(img)
            depth = self.depth_transform(depth)       
            
        img = torch.cat([img, depth], 0)  
        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname
        
    def __len__(self):
        return len(self.img_id)
        
class scaleNorm(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)


        return {'image': image, 'depth': depth}




class RandomCrop(object):
   
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
      
        i = random.randint(0, 20)
        j = random.randint(0, 20)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
         
        return {'image': image, 'depth': depth}


# Transforms on torch.*Tensor
class Normalize(object):
    def __init__(self, name):
        self.data_name= name
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        depth = depth / 255
     
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
        
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        if self.data_name=='PETA':
            depth = torchvision.transforms.Normalize(mean=[0.6425], std=[0.2979])(depth)
        elif self.data_name =='RAP':
            depth = torchvision.transforms.Normalize(mean=[0.8062], std=[0.2806])(depth)
        elif self.data_name == 'RAP2':
            depth = torchvision.transforms.Normalize(mean=[0.8250], std=[0.2620])(depth)
        elif self.data_name == 'PA100k':
            #pdb.set_trace()
            depth = torchvision.transforms.Normalize(mean=[0.695], std=[0.3022])(depth)
        
        
       
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth'], 
        image = image.transpose((2, 0, 1))
        #pdb.set_trace()
        depth = np.expand_dims(depth, 2).transpose((2, 0, 1)).astype(np.float)
        # Generate different label scales
       
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float()}

class MyPad(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth'] 

        image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=0)
        depth = cv2.copyMakeBorder(depth,10,10,10,10,cv2.BORDER_CONSTANT,value=0)
        
        sample['image'] = image
        sample['depth'] = depth

        return sample

def get_transform(args):
   
    #train_transform=transforms.Compose([scaleNorm(),MyPad(),RandomCrop(),RandomFlip(),ToTensor(),Normalize(args.dataset)])
                                                                   
                                                                   
    #val_transform=transforms.Compose([scaleNorm(), ToTensor(), Normalize(args.dataset)])

    train_transform = T.Compose([
        T.Resize((height, width)),
        #T.Pad(10),
        #T.RandomCrop((height, width)),
        #T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    val_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
        
    return train_transform, val_transform
