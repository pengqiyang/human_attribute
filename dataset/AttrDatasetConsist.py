import os
import pickle
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import pdb
from tools.function import get_pkl_rootpath
import torchvision.transforms as T
seg_transform_resize = T.Compose([
        T.Resize((256, 192)),
        T.ToTensor()
    ])

class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, transform_resize=None, target_transform=None):

        assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))
    
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.transform_resize = transform_resize
        self.target_transform = target_transform

        self.root_path = dataset_info.root
        #self.root_path = 'data/PETA_DPT'
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
        imgseg_path = os.path.join('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA_seg/', imgname)
        img = Image.open(imgpath)
        #pdb.set_trace()
        #img_mask = np.load('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA_mask/'+imgname.split('.')[0]+'.npy',allow_pickle=True)
        #mask_list = [i.cpu().detach().numpy() for i in (list(img_mask))]

        #print(np.stack(list(img_mask.cpu().detach().numpy())).shape)
        
        #mask_list = torch.from_numpy(np.array(mask_list)).float()
        
        #img_seg = Image.open(imgseg_path)
        
        img_f = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img_l = self.transform(img)
            #img_seg = seg_transform_resize(img_seg)
            img_s = self.transform_resize(img)
            img_lf = self.transform(img_f)
            img_sf = self.transform_resize(img_f)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img_l, img_s, img_lf, img_sf, gt_label,gt_label, imgname

    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    #normalize = T.Normalize(mean=[0.6425], std=[0.2979])
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        #T.Pad(10),
        #T.RandomCrop((height, width)),
        #T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    train_transform_resize = T.Compose([
        T.Resize((128, 96)),
        #T.Pad(10),
        #T.RandomCrop((height, width)),
        #T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    valid_transform_resize = T.Compose([
        T.Resize((128, 96)),
        T.ToTensor(),
        normalize
    ])


    return train_transform, train_transform_resize, valid_transform, valid_transform_resize
