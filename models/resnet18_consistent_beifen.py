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
part_init =[]

for i in range(35):
     part_init.append(torch.from_numpy(np.load('tools/layer3/'+str(i)+'.npy', allow_pickle=True)))
#pdb.set_trace()
#part_init = [i.numpy() for i in (list(part_init))]
part_init = torch.cat(part_init).unsqueeze(2).unsqueeze(3).cuda().float()
part_init = torch.nn.functional.normalize(part_init, p=2, dim=1, eps=1e-12, out=None)       
#part_init = torch.from_numpy(np.array(part_init)).float()


__all__ = ['resnet18_consistent']
total =[]
for i in range(35):
    total.append(torch.zeros(512).cuda())
sum_ = torch.zeros(35)


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)    # b * n * c
                z = torch.bmm(x_t, mu)      # b * n * k
                z = F.softmax(z, dim=2)     # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)       # b * c * k
                mu = self._l2norm(mu, dim=1)
                
        # !!! The moving averaging operation is writtern in train.py, which is significant.
        #pdb.set_trace()
        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu.matmul(z_t)                  # b * c * n
        x = x.view(b, c, h, w)              # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu, z_t.view(b,35,h,w)

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class gaussian_mask(nn.Module):
      

    def __init__(self, inplanes, planes,h ,w ,key  ):
        super(gaussian_mask, self).__init__()
        self.w = w
        self.h = h
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num=key
        self.keypoints_cov= nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.keypoints_bn =  nn.BatchNorm2d(inplanes)
        self.keypoints_fc_0 =  nn.Linear(inplanes, planes)
        self.keypoints_fc =  nn.Linear(planes, 3*key)
        self.keypoints_sig = nn.Sigmoid()#nn.LocalResponseNorm(2)
        
        self.pos_x = torch.zeros(h, w).cuda()
        self.pos_y = torch.zeros(h, w).cuda()

        for i in range(h):
            for j in range(w):
                self.pos_x[i][j] = i
                self.pos_y[i][j] = j
                
    def forward(self, x):
        keypoints = self.keypoints_cov(x)
        keypoints = self.keypoints_bn(keypoints)
        keypoints = self.avg_pool(keypoints)
        keypoints = self.keypoints_fc_0(keypoints.view(x.size()[0], -1))
        keypoints = self.keypoints_fc(keypoints)
        keypoints = self.keypoints_sig(keypoints)#B,3*K
        
        #mask = torch.zeros(x.size()[0], self.h, self.w).cuda()
        mask = []
      
        #pdb.set_trace()
        for k in range(self.num):
            #pdb.set_trace()
            temp_x = self.h*keypoints[:,3*k].unsqueeze(1).unsqueeze(2)#B,H,W
            temp_y = self.w*keypoints[:,3*k+1].unsqueeze(1).unsqueeze(2)
            temp_r = self.w*keypoints[:,3*k+2].unsqueeze(1).unsqueeze(2)
            #pdb.set_trace()
            #print(self.pos_x.expand((x.size()[0], x.size()[2], x.size()[3])).size())
            #print(temp_x.size())
            #B, W, H
            dis = torch.sqrt((self.pos_x.expand((x.size()[0], x.size()[2], x.size()[3]))- temp_x)**2+(self.pos_y.expand((x.size()[0], x.size()[2], x.size()[3]))-temp_y)**2)
            #mask = mask+ torch.exp(-dis/)
            temp = torch.exp(-dis/temp_r).unsqueeze(1)
            mask.append(F.interpolate(temp, size=[8, 6], mode="bilinear")  )
       
        return mask, self.h*keypoints 
          
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
    
class Att(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,nums, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Att, self).__init__()
       
        norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, nums)
        self.bn3 = norm_layer(nums)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sig = nn.Sigmoid()
       

    def forward(self, x):
       
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sig(out)
        #pdb.set_trace()
        #out = (out- torch.min(out.view(out.size()[0], -1), dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3)) /  (torch.max(out.view(out.size()[0], -1), dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3)- torch.min(out.view(out.size()[0], -1), dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3))



        return out, out
        
        
class LSTM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride):
        super(LSTM, self).__init__()
       
        norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes, outplanes, stride = stride)
        self.bn1 = norm_layer(outplanes)
        
        
        self.conv2 = conv1x1(inplanes, outplanes, stride = stride)
        self.bn2 = norm_layer(outplanes)
        
        self.conv3 = conv1x1(inplanes, outplanes, stride = stride)
        self.bn3 = norm_layer(outplanes)

        self.conv4 = conv1x1(inplanes, outplanes, stride = stride)
        self.bn4 = norm_layer(outplanes)
        
        self.tanh = nn.ReLU()
        self.sig = nn.Sigmoid()
       

    def forward(self, x_1, x_2):
       
        out_1 = self.conv1(x_1)
        out_1 = self.bn1(out_1)
        out_1 = self.tanh(out_1)

        out_2 = self.conv2(x_2)
        out_2 = self.bn2(out_2)
        out_2 = self.sig(out_2)
    
        out_3 = out_1*out_2    
   
        out_4 = self.conv3(x_2)
        out_4 = self.bn3(out_4)
        out_4 = self.sig(out_4)
        
        out_5 = out_3 + out_4
        
        out_6 = self.tanh(out_5) 
        
        out_7 = self.conv3(x_2)
        out_7 = self.bn3(out_7)
        out_7 = self.sig(out_7)
        
        out = out_7*out_6
        
        


        return out_5, out          



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
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2], temp=True)    
        self.conv_3 = conv1x1(256, 35)
        #self.conv_4 = conv1x1(512, 35)       

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

    def forward(self, x, img_mask, gt_label):
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
        out = self.conv_3(output_3)#BS , A ,H , W
        out = out.view(bs, 35, -1)
        #find max
        index = torch.where( ==  torch.max(out, dim=2))
        
        #get mask  #BS,A,H,W      
        
        

        return mask

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.fix = ResNet_fix(block, layers)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead

            # -----------------------------
            # modified
            replace_stride_with_dilation = [False, False, False]
            # -----------------------------

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        
        #self.conv2 = nn.Conv2d(64, 2, kernel_size=1, stride=1)
        #self.bn2 = norm_layer(2)        
        self.relu = nn.ReLU(inplace=True)
        
        self.sig = nn.Sigmoid()
        #self.lstm_1 = LSTM(64, 64, stride=1)
        #self.lstm_2 = LSTM(64, 128, stride=2)
        #self.lstm_3 = LSTM(128, 256, stride=2)
        #self.lstm_4 = LSTM(256, 512, stride=2)
        #self.spatial_att_0 = Att(inplanes=64, planes=32)
        #self.spatial_att_1 = Att(inplanes=64, planes=32)
        #self.spatial_att_2 = Att(inplanes=128, planes=32)
        #self.spatial_att_3 = Att(inplanes=512, planes=64, nums=6)
        #self.keypoints = Parameter(torch.Tensor(12,2))

        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], temp=True)    
        self.part_detector = nn.Conv2d(256, 103, kernel_size=1, stride=1, bias=False)
        self.up_dim = conv1x1(103, 512, stride=2)
        self.bn_2 = nn.BatchNorm2d(512)
        #self.relu_2 = nn.ReLU()

        #self.emau = EMAU(512, 35, 3)
        
        
        #nn.init.constant_(self.part_detector.weight, part_init)
        #)  
        #self.spatial_att_3 = 
        #self.gausion_att = gaussian_mask(512,512,8,6,7) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #self.part_detector.weight = nn.Parameter(part_init)
        #nn.init.constant_(self.part_detector.bias, 0)
        #for para in self.part_detector.parameters():
        #    para.requires_grad = False

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
        x = self.conv1(x)
        x = self.bn1(x)

        output_0 = self.relu(x)

        
        x = self.maxpool(output_0)
        
        
        
        '''
        
        mask = torch.ones(x.size()[0], 64, 48)
        for i in range(y_cov_1.size()[0]):
            input_mask = y_cov_1[i].reshape(15, -1 ).permute(1,0).reshape(-1,15)
            cluster_ids_x, cluster_centers = kmeans(X=input_mask, num_clusters=2, distance='euclidean', tqdm_flag=False, device=torch.device('cuda:0'))
            #pdb.set_trace()
            idx = cluster_ids_x.reshape(64, 48)
            #pdb.set_trace()
            mask[i] =  1 - (idx ^ (idx[32,24].unsqueeze(0).unsqueeze(1)))
            
        mask = mask.unsqueeze(1).float().cuda()
        
        
        y_cov_1 = torch.sum(y_cov_1, dim=1)/15
        #pdb.set_trace()
        #out = (y_cov_1- torch.min(y_cov_1.view(y_cov_1.size()[0], -1), dim=1)[0].unsqueeze(1).unsqueeze(2)) /  (torch.max(y_cov_1.view(y_cov_1.size()[0], -1), dim=1)[0].unsqueeze(1).unsqueeze(2)- torch.min(y_cov_1.view(y_cov_1.size()[0], -1), dim=1)[0].unsqueeze(1).unsqueeze(2))
        #pdb.set_trace()
        out = torch.tanh(y_cov_1)
        mask = out.unsqueeze(1)
        #mask = torch.nn.functional.softmax(y_cov_1.view(x.size()[0], -1), dim=1).view(x.size()[0], int(w/4), int(h/4)).unsqueeze(1)
        
        #mask =  self.spatial_att(x)
       
        #pdb.set_trace()
    

        
        #mask =  self.spatial_att_0(x)
       
        #pdb.set_trace()

        #mask = self.gausion_att(x)
        x = x*mask
        '''
        #x = x*F.interpolate(mask, size=[int(w/4), int(h/4)], mode="bilinear")
        output_1 = self.layer1(x)
    
        #hidden, out = self.lstm_1(x,x)
        
        #output_1 = output_1*F.interpolate(mask, size=[int(w/4), int(h/4)], mode="bilinear")
     
        
        #mask_1 =  self.spatial_att_1(output_1)
       
        output_2 = self.layer2(output_1)
        
        
        
        #hidden, out = self.lstm_2(hidden,output_1)
        #output_2 = output_2*F.interpolate(mask, size=[int(w/8), int(h/8)], mode="bilinear")
        
        #mask =  self.gausion_att(output_2)
        #mask_1 = self.spatial_att_2(output_2)
        output_3 = self.layer3(output_2)


        #hidden, out = self.lstm_3(hidden,output_2)
        #output_3 = output_3*F.interpolate(mask, size=[int(w/16), int(h/16)], mode="bilinear") 

        #mask, keypoints =  self.spatial_att_3(output_3)    
        
        #output_3 = torch.nn.functional.normalize(output_3, p=2, dim=1, eps=1e-12, out=None)
        #out = self.part_detector(output_3)
        #output_3 = torch.cat((out, output_3), dim=1)
        
        
        output_4 = self.layer4(output_3)
        

        #pdb.set_trace()
        #out = torch.nn.functional.normalize(out.view(bs, 105, -1), p=1, dim=2, eps=1e-12, out=None)
        #out = torch.max(out, dim=1)[0]
        #out =out.view(bs,16,12)
        #pdb.set_trace()
        
        #pdb.set_trace()
        #show_cam_on_image(out)
        #pdb.set_trace()
        #pdb.set_trace()
        #out = self.up_dim(output_3)
        #out = self.bn_2(out)
        #out = self.relu_2(out)
        #output_4 = out+output_4
        #pdb.set_trace()
        #img_mask = F.interpolate(img_mask, scale_factor=0.5, mode="bilinear")
        #output_4 = torch.cat((output_4, img_mask), dim=1)
        '''
        output=[]
        #pdb.set_trace()
        for i in range(35):
             
            out = output_3*img_mask[:,i,:,:].unsqueeze(1)
            output_4 = self.layer4(out)
            output.append(output_4)
        '''    
            
        
        #output_4, mu, feature_map = self.emau(output_4)
        
        #feature_map = self.avg_pool(output_4)
        #feature_map = feature_map.view(x.size()[0], -1)      
        
        #mask, keypoints =  self.spatial_att_3(output_4)    
        #hidden, out = self.lstm_4(hidden,output_3)    
        #mask = F.interpolate(mask, size=[int(w/32), int(h/32)], mode="bilinear")
        '''        
        for i in range(gt_label.size()[0]):
            channel = feature_map[i]#C
            sort, indices = torch.sort(channel, descending=True)
            for j in range(35):
                if gt_label[i][j]==1:
                    sum_[j]=sum_[j]+1
                    total[j][indices[:400]] =  total[j][indices[:400]] + 1
        for i in range(35):
        
            total[i] = total[i]/sum_[i]
            sort, indices = torch.sort(total[i], descending=True)
            np.save(str(i)+'.npy', indices[:100].cpu().detach().numpy() )       
            #output_4 = output_4+out
        '''
        return output_3, output_4, output_3, output_4#mask, keypoints


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
        temp = mask[i].cpu().detach().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * temp), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (192, 256))
        #print(img.shape, heatmap.shape)
        #heatmap = np.float32(heatmap) / 255

        #cam = 0.6*heatmap + 0.3*np.float32(img)

        #cam = cam / np.max(cam)
	
        cv2.imwrite("images/"+str(i)+'.jpg', np.uint8(heatmap)) 


