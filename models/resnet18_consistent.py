import torch.nn as nn
import cv2
import numpy as np
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torch.nn import Module, Parameter
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet
from .transformer import Transformer
import pdb
from kmeans_pytorch import kmeans
from models import MPNCOV
from models.High_Order import *
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import Adam
from models.cbam import CBAM
from models.cbam import *
from tools.nmf import NMF
from tools.utils_nmf import imresize, show_heatmaps
mask = torch.zeros(64,48)
from kmeans_pytorch import kmeans
from functools import partial
from bn_lib.nn.modules import SynchronizedBatchNorm2d
from torch.nn.modules.batchnorm import _BatchNorm
norm_layer = partial(SynchronizedBatchNorm2d, momentum=3e-4)
part_init_0 =[]
part_init_1 =[]
part_init_2 =[]
for i in range(35):
     part_init_0.append(torch.from_numpy(np.load('tools/layer3/'+str(i)+'.npy', allow_pickle=True))[0].unsqueeze(0))
     part_init_1.append(torch.from_numpy(np.load('tools/layer3/'+str(i)+'.npy', allow_pickle=True))[1].unsqueeze(0))
     part_init_2.append(torch.from_numpy(np.load('tools/layer3/'+str(i)+'.npy', allow_pickle=True))[2].unsqueeze(0))
#pdb.set_trace()
#part_init = [i.numpy() for i in (list(part_init))]
part_init_0 = torch.cat(part_init_0).cuda().float()
part_init_1 = torch.cat(part_init_1).cuda().float()
part_init_2 = torch.cat(part_init_2).cuda().float()
part_init_0 = torch.nn.functional.normalize(part_init_0, p=2, dim=1, eps=1e-12, out=None)#35, 256,1 
part_init_1 = torch.nn.functional.normalize(part_init_1, p=2, dim=1, eps=1e-12, out=None)#35, 256,1  
part_init_2 = torch.nn.functional.normalize(part_init_2, p=2, dim=1, eps=1e-12, out=None)#35, 256,1        
#part_init = torch.from_numpy(np.array(part_init)).float()



__all__ = ['resnet18_consistent']
total =[]
for i in range(35):
    total.append(torch.zeros(512).cuda())
sum_ = torch.zeros(35)
def generate_att_map(output_4, out_4):
        #out_4: bs, 35, h, w
        #output_4: bs, dim ,h, w
        
        #reshape the tensor
        bs = output_4.size()[0]
        h = output_4.size()[2]
        w = output_4.size()[3]         
        out = out_4.view(bs, 35, -1)
        #output_4 = output_4.view(bs, 512, -1)
        #find max
        #pdb.set_trace()
        
        output_4 = torch.nn.functional.normalize(output_4, p=2, dim=1, eps=1e-12, out=None)#BS, C, HW      
        
        att_map=[]
        max_val =  torch.max(out, dim=2)
        index  = max_val[0].unsqueeze(2).expand_as(out)#BS, 35, 192
        mask = (index == out)#BS, 35, HWx 
        output_4 = output_4.view(bs, -1, h*w)
        
        
        for i in range(35):
            #index = torch.where(out[i][j] ==  torch.max(out[i][j]))
            #x,y=index[0].data, index[1].data 
            #pdb.set_trace()   
            mask_temp = mask[:,i,:].unsqueeze(1).expand_as(output_4)#BS,256,HW
            #pdb.set_trace()
            #for j  in range (bs):
            # make only one response
            index =  torch.sum(mask_temp, dim=2)
            index =  torch.where(index>1)
            if index[0].size()[0]>1:
                #pdb.set_trace()
                for k in range(index[0].size()[0]):
                    #pdb.set_trace()
                    x = index[0][k]
                    y = index[1][k]
                    index_temp = torch.where(mask_temp[x,y,:]==True)
                    for label in range(1, index_temp[0].size()[0]):
                        mask_temp[x,y,index_temp[0][label]] = False
                    
            part = torch.masked_select(output_4, mask_temp.clone())#BS, 256 
            part = part.view(bs, -1).unsqueeze(2)#BS, 256, 1
            #print(part.size())
            #print(output_3.size())
            #part = torch.nn.functional.normalize(part, p=2, dim=1, eps=1e-12, out=None)
            mask_temp =  torch.matmul(output_4.permute(0,2,1), part)#BS, HW, 1
            att_map.append(mask_temp.view(bs, h, w).unsqueeze(0))
        
        att_map = torch.cat(att_map, dim=0) #35 ,bs , H,W
        return att_map
        
def max_activation(out):
     
     bs = out.size()[0]
     h = out.size()[2]
     w = out.size()[3]
     out = out.view(bs, 35, -1)
     max_val =  torch.argmax(out, dim=2, keepdim=True)
     #out = out.view(bs, 35, h, w)
     #index  = max_val[0].unsqueeze(2).unsqueeze(3).expand_as(out)#BS, 35, 192
     #ind = torch.where(index == out)#BS, 35, H;  BS , 35, W
     #x = ind[2].view(bs, -1).unsqueeze(2)
     #y = ind[3].view(bs, -1).unsqueeze(2)
     x = max_val/w
     y = max_val%w
     #pdb.set_trace()
     #x = ind[0].view(bs, -1).unsqueeze(2)
     #y = ind[1].view(bs, -1).unsqueeze(2)     
     return torch.cat([x, y], dim=2)
  
def sampling_max_feature(feature, ind):
    #ind： bs, 35, 2
    #feature: bs, 35, h, w
    #pdb.set_trace()
    bs = feature.size()[0]
    h = feature.size()[2]
    w = feature.size()[3]
    feat = []
    for i in range(35):
        #temp_t=[]
        #for j in range(35):
        #    x = ind[i, j, 0]
        #    y = ind[i, j, 1]
        #    #pdb.set_trace()
        #    temp_t.append(feature[i,:,x,y].unsqueeze(0))
        #pdb.set_trace()
        #feat.append(torch.cat(temp_t, dim=0).unsqueeze(0))
        #temp = torch.index_select(feature, 2, ind[:,i,0])# bs, dim, W
        temp_t = torch.gather(feature, 2, ind[:,i,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(feature))
        temp_t = torch.gather(temp_t, 3, ind[:,i,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(feature))
        #temp = torch.index_select(temp, 2, ind[:,i,1])
        feat.append(temp_t[:,:,0,0].unsqueeze(0))
    #pdb.set_trace()
    return torch.cat(feat, dim =0) #35,bs,dim
    
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


          
def norm(x, dim):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    normed = x / torch.sqrt(squared_norm)
    return normed
            
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_fix(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_fix, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)        
        self.relu = nn.ReLU(inplace=True)       
        self.sig = nn.Sigmoid()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], temp=True)        
        self.conv_4 = conv1x1(512, 35)
        self.conv_3 = conv1x1(256, 35)
        self.conv_2 = conv1x1(128, 35)
        self.conv_1 = conv1x1(64, 35)
        
        self.bn_1 = nn.BatchNorm1d(35)
        self.bn_2 = nn.BatchNorm1d(35)
        self.bn_3 = nn.BatchNorm1d(35)
        self.bn_4 = nn.BatchNorm1d(35)
        #self.keypoints_0 = Parameter(torch.Tensor(35,256))      
        #self.keypoints_1 = Parameter(torch.Tensor(35,256))  
        #self.keypoints_2 = Parameter(torch.Tensor(35,256))          
        #self.part_detector = nn.Conv2d(256, 105, kernel_size=1, stride=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, temp=False):
        if temp==True:
            self.inplanes=256
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, gt_label):
        bs = x.size()[0]
        w = x.size()[2]
        h = x.size()[3]
        x = self.conv1(x)
        x = self.bn1(x)

        output_0 = self.relu(x)       
        x = self.maxpool(output_0)        
        output_1 = self.layer1(x)            
        output_2 = self.layer2(output_1)
        output_3 = self.layer3(output_2)  
        output_4 = self.layer4(output_3)        
        
        out_4 = self.conv_4(output_4)#BS , A ,H , W
        out_3 = self.conv_3(output_3)
        out_2 = self.conv_2(output_2)
        out_1 = self.conv_1(output_1)
        #pdb.set_trace()
        re_3  = self.bn_3(self.max_pool(out_3).view(bs, 35))
        re_4  = self.bn_4(self.max_pool(out_4).view(bs, 35))
        re_2  = self.bn_2(self.max_pool(out_2).view(bs, 35))
        re_1  = self.bn_1(self.max_pool(out_1).view(bs, 35))
        #max_loc_2 = max_activation(out_2)
        max_loc_2 = max_activation(out_2)
        max_loc_3 = max_activation(out_3)
        max_loc_4 = max_activation(out_4)
        
        
        
        '''
        
        #att_map_0 = torch.matmul(output_3.permute(0,2,1), part_init_0.permute(1,0))#BS, HW, 35
        #att_map_1 = torch.matmul(output_3.permute(0,2,1), part_init_1.permute(1,0))
        #att_map_2 = torch.matmul(output_3.permute(0,2,1), part_init_2.permute(1,0))
        #pdb.set_trace()
        #att_map =  torch.max(torch.max(att_map_1, att_map_0), att_map_2) 
        #mask = self.part_detector(output_3)
        '''
        '''
        for i in range(35):
            mask = self.part_detector(output_3)
            mask =(output_3 * (part_init[i].squeeze().unsqueeze(0).unsqueeze(2).expand_as(output_3)))#35, 256, HW
            mask = torch.sum(mask, dim=1)#35,HW
            att_map.append(mask.view(bs, 16,12).unsqueeze(0))
        '''
        #att_map = torch.cat(att_map, dim=0) #35 ,bs , H,W
        #att_map = att_map.permute(0,2,1).view(bs, 35, 16, 12)
        return  max_loc_2, max_loc_3, max_loc_4
        #pdb.set_trace()
        
        #return mask#BS, 105, H ,W       
        
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        #self.fix = ResNet_fix(block, layers)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:    
            replace_stride_with_dilation = [False, False, False]
      
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], temp=True)    
        '''
        self.part_detector = nn.Conv2d(256, 103, kernel_size=1, stride=1, bias=False)
        self.up_dim = conv1x1(256, 512, stride=1)
        self.bn_2 = nn.BatchNorm2d(512)
        self.cbam = SpatialGate()
        '''
        
        self.conv_4 = conv1x1(512, 35)
        self.conv_3 = conv1x1(256, 35)
        self.conv_2 = conv1x1(128, 35)
        self.conv_1 = conv1x1(64, 35)
        
        self.bn_1 = nn.BatchNorm1d(35)
        self.bn_2 = nn.BatchNorm1d(35)
        self.bn_3 = nn.BatchNorm1d(35)
        self.bn_4 = nn.BatchNorm1d(35)        
        
        
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #self.fix.part_detector.weight = nn.Parameter(part_init)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, temp=False):
        if temp==True:
            self.inplanes=256
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, img_mask, gt_label):
        bs = x.size()[0]
        w = x.size()[2]
        h = x.size()[3]
        #loc_2, loc_3, loc_4 = self.fix(x, gt_label)#BS, A, H ,W 
  
        x = self.conv1(x)
        x = self.bn1(x)
        output_0 = self.relu(x)
        x = self.maxpool(output_0)
        output_1 = self.layer1(x)
        output_2 = self.layer2(output_1)
        output_3 = self.layer3(output_2)
        output_4 = self.layer4(output_3)
        
        
        out_4 = self.conv_4(output_4)#BS , A ,H , W
        out_3 = self.conv_3(output_3)
        out_2 = self.conv_2(output_2)
        out_1 = self.conv_1(output_1)

       
        loc_1 = max_activation(out_1)
        loc_2 = max_activation(out_2)
        loc_3 = max_activation(out_3)
        loc_4 = max_activation(out_4)
        
        re_1  = self.bn_3(self.max_pool(out_1).view(bs, 35))
        re_3  = self.bn_3(self.max_pool(out_3).view(bs, 35))
        re_4  = self.bn_4(self.max_pool(out_4).view(bs, 35))
        re_2  = self.bn_2(self.max_pool(out_2).view(bs, 35))
         
        #att_map  = generate_att_map(output_3, out_3)        
        
        feature_1 = sampling_max_feature(output_1, loc_1)#35 ,bs, dim
        feature_2 = sampling_max_feature(output_2, loc_2)#35 ,bs, dim
        feature_3 = sampling_max_feature(output_3, loc_3)#35 ,bs, dim     
        feature_4 = sampling_max_feature(output_4, loc_4)#35 ,bs, dim
       
        #pdb.set_trace()
        
        '''
        output_3 = output_3.unsqueeze(0)#1, BS, C, H,W
        
        mask = mask_temp.unsqueeze(2)#A,BS, 1,H, W
        
        local = output_3*mask #A, BS, C , H, W 
        local = local.reshape(35*bs, 256, 16, 12)
        local = self.max_pool(local)#A*BS,C,1,1
        local = local.reshape(35, bs, 256,1)#A, BS, 256, 1
        local = local.permute(1,2,0,3)#BS,C , A,1
        local = self.cbam(local)
        local = self.up_dim(local)#BS,512,A,1
        #spatial attention        
        local = self.avg_pool(local)#bs, 512, 1,1
        '''
        
        
        
        return output_4, feature_1, feature_2, feature_3, feature_4, loc_3, re_2, re_3, re_4
        #return re_1, re_2, re_3, re_4#mask, keypoints


def remove_fc(state_dict):
    """ Remove the fc layer parameter from state_dict. """
    return {key: value for key, value in state_dict.items() if not key.startswith('fc.')}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    '''
    model = ResNet(block, layers, **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(remove_fc(state_dict))
    return model
    '''
    model = ResNet(block, layers, **kwargs)
    
    #pretrain from the model   version1:
    pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    #pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    
    return model

def resnet18_consistent(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
                   
def show_cam_on_image(mask):
    pdb.set_trace()
    for i in range(35):
        '''
        #pdb.set_trace()
        mask_1 = torch.sum(mask[0]*(part_init[i*3].expand_as(mask[0])), dim=0)
        mask_2 = torch.sum(mask[0]*(part_init[i*3+1].expand_as(mask[1])), dim=0)
        mask_3 = torch.sum(mask[0]*(part_init[i*3+2].expand_as(mask[2])), dim=0)
        temp = torch.max(torch.max(mask_1, mask_2), mask_3)
        '''
        temp = mask[i,0,:,:].cpu().detach().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * temp), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (192, 256))
        #print(img.shape, heatmap.shape)
        #heatmap = np.float32(heatmap) / 255

        #cam = 0.6*heatmap + 0.3*np.float32(img)

        #cam = cam / np.max(cam)
	
        cv2.imwrite("images/"+str(i)+'.jpg', np.uint8(heatmap)) 


