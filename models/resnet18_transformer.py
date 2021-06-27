import torch.nn as nn
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet
from .transformer import Transformer
import pdb
from models import MPNCOV
from models.High_Order import *
__all__ = ['resnet18_transformer']

            
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
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.att_conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.transformer_0 = Transformer(num_layers=1, dim=128, h=32, w=24, num_heads=1, 
                                      ff_dim=128, dropout=0.1)
        self.transformer_1 = Transformer(num_layers=1, dim=256, h=16, w =12, num_heads=1, 
                                      ff_dim=256, dropout=0.1)
        self.transformer_2 = Transformer(num_layers=1, dim=512, h=8, w=6, num_heads=1, 
                                      ff_dim=512, dropout=0.1)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes) 
        #self.fc_1 = nn.Linear(512, 128)
        #self.fc_2 = nn.Linear(128, 512)
        #self.sig = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.soca_0  = SOSA(64, reduction=8)
        #self.soca_1  = SOSA(64, reduction=8)
        #self.soca_2  = SOSA(64, reduction=8)
        #self.somm = SOMM(64, 64, 48)
        #self.somm_0 = SOMM(128, 32, 24)
        #self.somm_1 = SOMM(256, 16, 12)
        #self.somm_2 = SOMM(512, 8, 6)
        #self.sonl_0 = SONL(128)
        #self.sonl_1 = SONL(256)
        #self.sonl_2 = SONL(512)
        #self.NLB_0 = NLBlockND(in_channels=256, dimension=2)
        #self.NLB_1 = NLBlockND(in_channels=512, dimension=2)
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
        x = self.relu(x)
        

        
        #pdb.set_trace()
        
        
        
        x = self.maxpool(x)
        
        #pdb.set_trace()

        x = self.layer1(x)
       
        #x= self.somm(x)
        
        #x = self.sig(x)

        x = self.layer2(x)
        x = self.transformer_0(x.permute(0,2,3,1).view(x.size()[0], x.size()[2]*x.size()[3], 128))
        x = x.permute(0,2,1).view(x.size()[0], 128, 32, 24)
        #x = self.somm_0(x)
        #pdb.set_trace()
        #x = self.sig(x)
        #x = self.soca_0(x)
       
        x = self.layer3(x)
        x = self.transformer_1(x.permute(0,2,3,1).view(x.size()[0], x.size()[2]*x.size()[3], 256))
        x = x.permute(0,2,1).view(x.size()[0], 256, 16, 12)
        #x = self.somm_1(x)
        #x = self.NLB_0(x)
        #x = self.NLB_0(x)
        
        x = self.layer4(x)
        x = self.transformer_2(x.permute(0,2,3,1).view(x.size()[0], x.size()[2]*x.size()[3], 512))
        x = x.permute(0,2,1).view(x.size()[0], 512, 8, 6)
        #x = self.somm_2(x)
        #x = self.NLB_1(x)
        
        #x = self.soca_2(x)
        #bs, dim, w, h =x.size()[0], x.size()[1], x.size()[2], x.size(3)
        #x = x.view(x.size()[0], x.size()[1], -1).permute(0, 2, 1)
        #x = self.transformer_1(x)
        #x = x.permute(0,1,2).view(bs, dim, w, h)
        #att = self.avg_pool(x).view(x.size()[0], x.size(1))
        #pdb.set_trace()
        #att = self.fc_1(att)
        #att = self.relu(att)

        #att = self.fc_2(att)
        #att = self.sig(att)
        #att = torch.mean(att, dim=0).unsqueeze(0)
        
        #x = x*att.unsqueeze(2).unsqueeze(3)
        return x


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

def resnet18_transformer(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)




if __name__ == '__main__':
    # print(resnet50())
    model = resnet50().cuda()
    x = torch.rand((1, 3, 256, 128)).cuda()
    model(x)

    # print('receptive_field_dict')
