import torch.nn as nn
import pdb
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet

__all__ = ['resnet18_self_mutual_attention']

class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv2d

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.ImageAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.DepthAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)

    def forward(self, self_fea, mutual_fea, alpha, selfImage):

        if selfImage:
            #pdb.set_trace()
            selfNonLocal_fea = self.ImageAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.DepthAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # using mutual feature to generate attention
            theta_x = self.F_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            # using self feature to generate attention
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            # add self_f and mutual f
            f_div_C = F.softmax(alpha*f + self_f, dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.R_W(y)
            z = W_y + self_fea
            return z

        else:
            selfNonLocal_fea = self.DepthAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.ImageAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.F_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # using mutual feature to generate attention
            theta_x = self.R_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.R_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            # using self feature to generate attention
            self_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.F_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            # add self_f and mutual f
            f_div_C = F.softmax(alpha*f+self_f, dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.F_W(y)
            z = W_y + self_fea
            return z
            
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
        inplace  = True
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
        
        #layer for merge
        # S2MA module
        self.NonLocal_0 = NonLocalBlock(in_channels=64)
        self.NonLocal_1 = NonLocalBlock(in_channels=64)
        self.NonLocal_2 = NonLocalBlock(in_channels=128)
        
        self.image_bn_relu_0 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.depth_bn_relu_0 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
            
        self.image_bn_relu_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.depth_bn_relu_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
            
        self.image_bn_relu_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.depth_bn_relu_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.affinityAttConv_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )

        self.affinityAttConv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        
        self.affinityAttConv_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )        
        #layer for depth layer
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_d = norm_layer(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_d = self._make_layer(block, 64, 1)
        self.layer2_d = self._make_layer(block, 128, 1, stride=2,
                                       dilate=replace_stride_with_dilation[0])

        
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

    def forward(self, x, depth):
            
        pdb.set_trace()  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        
        #network for depth
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu_d(depth)
        depth = self.maxpool_d(depth)   
        # fusion 0
        
        bs, ch, hei, wei = depth.size()

        affinityAtt = F.softmax(self.affinityAttConv_0(torch.cat([x, depth], dim=1)))
        alphaD = affinityAtt[:, 0, :, :].reshape([bs, hei * wei, 1])
        alphaR = affinityAtt[:, 1, :, :].reshape([bs, hei * wei, 1])

        alphaD = alphaD.expand([bs, hei * wei, hei * wei])
        alphaR = alphaR.expand([bs, hei * wei, hei * wei])

        ImageAfterAtt1 = self.NonLocal_0(x, depth, alphaD, selfImage=True)
        DepthAfterAtt1 = self.NonLocal_0(depth, x, alphaR, selfImage=False)

        x = self.image_bn_relu_0(ImageAfterAtt1)
        depth = self.depth_bn_relu_0(DepthAfterAtt1)
        
  
       
                     
        x = self.layer1(x)   
        depth = self.layer1_d(depth)       
        #fusion 1
        bs, ch, hei, wei = depth.size()

        affinityAtt = F.softmax(self.affinityAttConv_1(torch.cat([x, depth], dim=1)))
        alphaD = affinityAtt[:, 0, :, :].reshape([bs, hei * wei, 1])
        alphaR = affinityAtt[:, 1, :, :].reshape([bs, hei * wei, 1])

        alphaD = alphaD.expand([bs, hei * wei, hei * wei])
        alphaR = alphaR.expand([bs, hei * wei, hei * wei])

        ImageAfterAtt1 = self.NonLocal_1(x, depth, alphaD, selfImage=True)
        DepthAfterAtt1 = self.NonLocal_1(depth, x, alphaR, selfImage=False)

        x = self.image_bn_relu_1(ImageAfterAtt1)
        depth = self.depth_bn_relu_1(DepthAfterAtt1)        

         
        x = self.layer2(x)
        depth = self.layer2_d(depth)
        #fusion 2
        bs, ch, hei, wei = depth.size()

        affinityAtt = F.softmax(self.affinityAttConv_2(torch.cat([x, depth], dim=1)))
        alphaD = affinityAtt[:, 0, :, :].reshape([bs, hei * wei, 1])
        alphaR = affinityAtt[:, 1, :, :].reshape([bs, hei * wei, 1])

        alphaD = alphaD.expand([bs, hei * wei, hei * wei])
        alphaR = alphaR.expand([bs, hei * wei, hei * wei])

        ImageAfterAtt1 = self.NonLocal_2(x, depth, alphaD, selfImage=True)
        #DepthAfterAtt1 = self.NonLocal_2(depth, x, alphaR, selfImage=False)

        x = self.image_bn_relu_2(ImageAfterAtt1)
        #depth = self.depth_bn_relu_2(DepthAfterAtt1)                
        
        
        x = self.layer3(x)   
        x = self.layer4(x)

       
        return x


def remove_fc(state_dict):
    """ Remove the fc layer parameter from state_dict. """
    return {key: value for key, value in state_dict.items() if not key.startswith('fc.')}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #pretrain from the model   version1:
    
    pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    #pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    '''
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

            elif k.startswith('bn1'):
                model_dict[k] = v
                model_dict[k.replace('bn1', 'bn1_d')] = v
            elif k.startswith('layer1.0.'):
                model_dict[k] = v
                model_dict[k[:6]+'_d'+k[6:]] = v  
            elif k.startswith('layer2.0.'):
                model_dict[k] = v
                model_dict[k[:6]+'_d'+k[6:]] = v 
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    '''
    return model


def resnet18_self_mutual_attention(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


