import torch.nn as nn
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
mask = torch.zeros(64,48)

__all__ = ['spatial_modulator']
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
    
    
def spatial_optimize(fmap):
    bs = fmap.size()[0]
    cur_fmap = fmap
    with torch.no_grad():
        spatial_x = cur_fmap.permute(0, 2, 3, 1).contiguous().view(-1, cur_fmap.size(1)).transpose(1, 0)
        spatial_x = norm(spatial_x, dim=0)
        spatial_x_t = spatial_x.transpose(1, 0)
        G = spatial_x_t @ spatial_x - 1
        G = G.detach().cpu()

    with torch.enable_grad():
        spatial_s = nn.Parameter(torch.sqrt((bs*48) * torch.ones(((bs*48), 1))) / (bs*48), requires_grad=True)
        spatial_s_t = spatial_s.transpose(1, 0)
        spatial_s_optimizer = Adam([spatial_s], 0.01)

        for iter in range(200):
            f_spa_loss = -1 * torch.sum(spatial_s_t @ G @ spatial_s)
            spatial_s_d = torch.sqrt(torch.sum(spatial_s ** 2))
            if spatial_s_d >= 1:
                d_loss = -1 * torch.log(2 - spatial_s_d)
            else:
                d_loss = -1 * torch.log(spatial_s_d)

            all_loss = 50 * d_loss + f_spa_loss

            spatial_s_optimizer.zero_grad()
            all_loss.backward()
            spatial_s_optimizer.step()

    result_map = spatial_s.data.view(bs, 1, 8, 6)

  
    spa_mask = result_map
        
    return spa_mask


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

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
        #self.att_conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.cbam_0 = CBAM(64, reduction_ratio=16, pool_types=['crow'], no_spatial=False)
        #self.cbam_1 = CBAM(128, reduction_ratio=16, pool_types=['crow'], no_spatial=False)
        #self.cbam_2 = CBAM(256, reduction_ratio=16, pool_types=['crow'], no_spatial=False)
        #self.cbam_3 = CBAM(512, reduction_ratio=16, pool_types=['crow'], no_spatial=False)
        '''
        self.down_0 = nn.Conv2d(64, 128, kernel_size=3,stride=2, padding=1)
        self.in_cha_0 = nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1)
        self.down_1 = nn.Conv2d(128, 256, kernel_size=3,stride=2, padding=1)
        self.in_cha_1 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1)
        self.down_2 = nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1)
        self.in_cha_2 = nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1)
        '''     
        '''
        self.conv_du_0 = nn.Sequential(
            nn.Conv2d(16, 16 // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16 // 4, 16, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )
        '''
        '''
        self.conv_du_2 = nn.Sequential(
            nn.Conv2d(512, 512 // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // 4, 512, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )
        '''
        #self.cov = SOMM_v3(11)
        #self.cov = nn.Conv2d(64, 1, 1, padding=0, bias=False)
        #self.bn_att = norm_layer(1)
        
        
        #self.selective_0 = Parameter(torch.Tensor(1))
        
        #self.spa_conv_0 = nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1)
        #self.spa_bn_0 = nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True)
        '''        
        self.spa_conv_2 = nn.Conv2d(512, 1, kernel_size=3, stride=1,padding=1)
        self.spa_bn_2 = nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True) 
        '''
        #nn.init.constant_(self.selective_0, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        output_0 = self.relu(x)
        #pdb.set_trace()
      
        #crow_pool = gem(output_0)
        #pdb.set_trace()
        
        x = self.maxpool(output_0)
        #x = norm(x, dim=1)
        #version -1: CBAM
        mask = torch.ones(x.size()[0], 64, 48)
        for i in range(x.size()[0]):
            input_mask = x[i].reshape(64, -1 ).permute(1,0).reshape(-1,64)
            cluster_ids_x, cluster_centers = kmeans(X=input_mask, num_clusters=2, distance='euclidean', tqdm_flag=False, device=torch.device('cuda:0'))

            idx = cluster_ids_x.reshape(64, 48)
            #pdb.set_trace()
            mask[i] =  1 - (idx ^ (idx[32,24].unsqueeze(0).unsqueeze(1)))
            
        mask = mask.unsqueeze(1).float().cuda()
        
        
        
        #version 2  gem+SAM 
        
        
        
        
        #version 3 CROW+SAM
       
        # version0： constrain on feature map
        '''
        att = self.compress(output_0)
        att = self.cov(att)
        #pdb.set_trace()
        att = self.relu(att)
        #mask = self.sig(att)
        mask = torch.where(att>self.selective_0, torch.FloatTensor([1.0]).cuda(),torch.FloatTensor([0.0]).cuda())
        '''
        #x  = self.cbam_0(x)
        output_1 = self.layer1(x)
        
       
        # version1： generate the mask 
        '''
        y_cov_1 = 0 
        for i in [0,60,3,12,19,22,15,29,45,35,50,51,54,55,58,62]:
            y_cov_1 = y_cov_1 + output_0[:,i,:,:]
        y_cov_2 = torch.sum(output_0, dim =1) - y_cov_1    
        y_cov = (y_cov_1/16).unsqueeze(1) #* self.selective_0    
       
        y_cov = torch.tanh(y_cov)
   
        mask = y_cov.view(y_cov.size()[0], 1, 128,96)
        '''
        
        # version2： NLB generate the mask
        
        '''
        y_cov_1 = [] 
        for i in [3,12,29,35,54,58,62]:
            y_cov_1.append((output_0[:,i,:,:]))
               
        
        #pdb.set_trace()
        y_cov_1 = torch.stack(y_cov_1, dim=1)
        y_cov_1 = y_cov_1 * y_cov_1
        y_cov_1 = torch.sum(y_cov_1, dim=1)
        mask  = y_cov_1/torch.sum(y_cov_1.view(y_cov_1.size()[0], -1), dim=1).unsqueeze(1).unsqueeze(2)
        mask = mask.view(mask.size()[0], 1, 128,96)
        '''
        '''
        y_cov_1 = self.cov(y_cov_1)  
        y_cov_1 = torch.sum(y_cov_1, dim =1)        
        y_cov = (y_cov_1/11).unsqueeze(1) #* self.selective_0           
        y_cov = torch.tanh(y_cov)   
        '''
        '''
        x_compress = self.compress(y_cov_1)
        x_out = self.cov(x_compress)
        mask = torch.tanh(x_out) 
        
        mask = mask.view(mask.size()[0], 1, 128,96)
        '''

        
        #x_cov = F.softmax(x_cov.view(x_cov.size()[0], 1, -1), dim=-1)
        output_1 = output_1*F.interpolate(mask, size=[64, 48], mode="bilinear") +  output_1

        output_2 = self.layer2(output_1)
        #output_2  = self.cbam_1( output_2)
        output_2 = output_2*F.interpolate(mask, size=[32, 24], mode="bilinear") + output_2
        
        
        output_3 = self.layer3(output_2)
        #output_3  = self.cbam_2(output_3)
        output_3 = output_3*F.interpolate(mask, size=[16, 12], mode="bilinear") + output_3

        output_4 = self.layer4(output_3)
        #output_4  = self.cbam_3(output_4)
        output_4 = output_4*F.interpolate(mask, size=[8, 6], mode="bilinear") + output_4
        
         
              
        '''
        att = self.down_0(output_1)+self.in_cha_0(output_2)
        att = self.down_1(att)+self.in_cha_1(output_3)
        att = self.down_2(att)+self.in_cha_2(output_4)         
        spa_mask = spatial_optimize(att).cuda()
        
        x = output_4*spa_mask
        '''
        

        
        
        return output_4, output_4, mask#x_cov.view(x_cov.size()[0], 1, 128,96), output_4


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

def spatial_modulator(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



