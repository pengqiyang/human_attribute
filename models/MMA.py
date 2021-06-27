import torch
import torch.nn.functional as F
#import mxnet as mx
import numpy as np
import pdb
import torch.nn as nn
lg = nn.LogSigmoid()
mse_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
import torch.nn.functional as F
fuxiangguan = torch.from_numpy(np.load('tools/fuxiangguan.npy')).cuda().float() 
xiangguan = torch.from_numpy(np.load('tools/xiangguan.npy')).cuda().float() 
nonxiangguan = torch.from_numpy(np.load('tools/nonxiangguan.npy')).cuda().float() 
#nonxiangguan = torch.from_numpy(np.load('tools/nonxiangguan.npy')).cuda()
gt = torch.tensor(-0.99).cuda().float()
def get_mma_loss(weight):
   
    #pdb.set_trace()
    # for convolutional layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # computing cosine similarity: dot product of normalized weight vectors
    weight = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight, weight.t())

    # make sure that the diagnonal elements cannot be selected
    #cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    #cosine_uncorre = cosine[uncorre_mask == 1]
    #cosine_corre = cosine[corre_mask == 1]
    #agular
    fuxiangguan_cosine = cosine[fuxiangguan==1]
    xiangguan_cosine = cosine[xiangguan==1]
    nonxiangguan_cosine = cosine[nonxiangguan==1]
    #cc = gt.expand_as(cosine_select)
    #loss = F.mse_loss(cosine_select, cc)
    # maxmize the minimum angle of the uncorre
    loss_1 = -torch.acos(fuxiangguan_cosine.max().clamp(-0.99999, 0.99999)).mean()#最大化 负相关属性的最小夹角
    loss_2 = torch.acos(xiangguan_cosine.min().clamp(-0.99999, 0.99999)).mean()    #最小化，正相关属性的最大夹角
    #loss_3 = F.mse_loss(nonxiangguan_cosine, torch.tensor(0).expand_as(nonxiangguan_cosine).cuda().float())                                                              #不相关属性的夹角正交 
    #loss_1 = -torch.acos(cosine_uncorre.max().clamp(-0.99999, 0.99999)).mean() 
    #  minmize the maxmize angle of the corre attr
    #loss_2 = torch.acos(cosine_corre.min().clamp(-0.99999, 0.99999)).mean()
    #loss_2 = cosine_corre.min().clamp(-0.99999, 0.99999)).mean()
    
    
    # log sifmoid
    #loss = lg(cosine_corre.min())
    #return loss_1 + loss_2
    return 0.3*loss_1+ 0.3*loss_2# +loss_3
    
def get_feat_loss(feature_map, gt_label):
    
    '''
    feature_map = (feature_map*gt_label.unsqueeze(2).unsqueeze(3)).view(feature_map.size(0), feature_map.size(1),-1, -1)#BS, C -,1
    feature_map_ = F.normalize(feature_map, p=2, dim=2)
    cosine = torch.matmul(feature_map_, feature_map_.transpose(0,2,1)) #BS,C,C
    xiangguan_feat = cosine*xiangguan.unsqueeze(0).view(feature_map.size(0), -1)
    nonxiangguan_feat = cosine*nonxiangguan.unsqueeze(0).view(feature_map.size(0), -1)
  
    loss_1 = -torch.acos(nonxiangguan_feat.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean() 
    loss_2 = torch.acos(xiangguan_feat.min(dim=1)[0].clamp(-0.99999, 0.99999)).mean()   
    '''
    #pdb.set_trace()
    mask = (gt_label.unsqueeze(2))*(gt_label.unsqueeze(2).permute(0,2,1))
    #feature_map = ((feature_map)>0).float()
    feature_map =  feature_map.view(feature_map.size()[0], 35, -1)
    intersection = torch.matmul(feature_map, feature_map.permute(0,2,1))#BS, 35,35
    unique = torch.sum((feature_map * feature_map), dim=2).unsqueeze(2) #BS , 35,1 
    
    IOU = intersection/(unique+0.0001)
    ground_truth = gongxian.unsqueeze(0).expand_as(IOU)*mask
    #pdb.set_trace()
    loss = mse_loss_fn(ground_truth, IOU*mask)
    return loss   
    
    