import torch.nn as nn
import pdb
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet

__all__ = ['resnet_attention_depth_cbam_spatial']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(channel, channel // reduction, bias=False)
        self.re_1 = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(channel // reduction, channel, bias=False)
        self.re_2 = nn.ReLU(inplace=True)
        self.si = nn.Sigmoid()
	            

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
       
        y = self.fc_1(y)
        y = self.re_1(y)
        y = self.fc_2(y)
        y = self.re_2(y)
        y = self.si(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
        
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)        
class SBAM(nn.Module):
    def __init__(self):
        super(SBAM, self).__init__()
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.SpatialGate(x)
        return x_out
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
        return  scale       
        
        
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
        self.base_width = width_per_group
        #layer  for RGB input 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        #layer for spatial attention
        self.spatial_0 = SBAM()
        #self.spatial_1 = SBAM()
        #self.spatial_2 = SBAM()

        
        
        #layer for merge
        '''
        self.att_d = SELayer(64)
        self.att_rgb = SELayer(64)

        self.att_d_layer1 = SELayer(64)
        self.att_rgb_layer1 = SELayer(64)
        
        self.att_d_layer2 = SELayer(128)
        self.att_rgb_layer2 = SELayer(128)
        
        self.att_d_layer3 = SELayer(256)
        self.att_rgb_layer3 = SELayer(256)
        
        self.att_d_layer4 = SELayer(512)
        self.att_rgb_layer4 = SELayer(512)
        '''
        
       #layer for depth layer
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_d = norm_layer(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        '''
        self.layer1_d = self._make_layer(block, 64, layers[0])
        
        
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        '''
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

    def forward(self, x, true_depth):
        batch_size = x.size()[0]

       

        
        #pdb.set_trace()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #output_rgb_0 = x
   
        depth = self.conv1_d(true_depth)
        depth = self.bn1_d(depth)
        depth = self.relu_d(depth)
        depth = self.maxpool_d(depth)
        
        #output_0 = F.sigmoid(F.interpolate(true_depth, size=[64, 48]))
        #true_depth = depth    
        output_0 = self.spatial_0(depth)
        x = x + x * output_0
      


        
        x = self.layer1(x)
        #depth = self.layer1_d(depth)
        #output_1 = self.spatial_1(depth)
        #x = x + x * output_1
        #depth = self.layer1_d(depth) 
        #output_rgb_1 = x
        #output_depth_1 = depth     
        #output_fusion_1 = x 
        #x = self.att_d_layer1(depth)  + self.att_rgb_layer1(x) #merge        
        #output_1 = self.spatial_1(depth)
        #x = x + x * output_1

        x = self.layer2(x)
        #depth = self.layer2_d(depth) 
        
        #x = self.att_d_layer2(depth) + self.att_rgb_layer2(x) #merge
        #output_2 = self.spatial_2(depth)
        #x = x + x * output_2
       

        
        x = self.layer3(x)
        #depth = self.layer3_d(depth) 
        #x = self.att_d_layer3(depth) + self.att_rgb_layer3(x) #merge
        #output_depth_3 = depth

        x = self.layer4(x)
        #depth = self.layer4_d(depth)
        
        #x = self.att_d_layer4(depth) + self.att_rgb_layer4(x) #merge        
        

       
        return x, output_0


def remove_fc(state_dict):
    """ Remove the fc layer parameter from state_dict. """
    return {key: value for key, value in state_dict.items() if not key.startswith('fc.')}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #pretrain from the model   version1:
    pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        # print('%%%%% ', k)
        if k in state_dict:
            if k.startswith('conv1'):
                model_dict[k] = v
                # print('##### ', k)
                model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                    view_as(state_dict[k.replace('conv1', 'conv1_d')])
                #model_dict[k.replace('conv1', 'conv1_d')]  = v
            elif k.startswith('bn1'):
                model_dict[k] = v
                model_dict[k.replace('bn1', 'bn1_d')] = v
            
            elif k.startswith('layer'):
                model_dict[k] = v
            '''    
            elif k.startswith('layer1'):
                model_dict[k] = v
                model_dict[k[:6]+'_d'+k[6:]] = v
            '''
               
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    '''
    #pre-trained from the model version2
 
    pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    #pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    '''
    return model


def resnet_attention_depth_cbam_spatial(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


