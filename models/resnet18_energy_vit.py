import torch.nn as nn
import pdb
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from .transformer_v3 import Transformer
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet
from torch.autograd import Variable
from torch.autograd import Function

__all__ = ['resnet18_energy_vit']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}
class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
                                       
        #self.layer5 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=3,
        #                       bias=False)
        #self.conv2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, 
        #                       bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

     

        self.transformer_2 = Transformer(num_layers=1, dim=128, num_heads=4, 
                                       ff_dim=256, dropout=0.1)
        self.transformer_3 = Transformer(num_layers=1, dim=256, num_heads=4, 
                                       ff_dim=512, dropout=0.1)
        self.transformer_4 = Transformer(num_layers=1, dim=512, num_heads=4, 
                                      ff_dim=1024, dropout=0.1)                             
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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
        x = self.maxpool(x)
        


        x = self.layer1(x)
        x = Energy_Function_Max(x)
   
   
   
        
        x = self.layer2(x)   
        spatial_att2 = self.transformer_2(x.permute(0, 2, 3, 1).view(x.size()[0], x.size()[2]*x.size()[3], 128))
        x = x + spatial_att2.transpose(2, 1).view(x.size()[0], 128, x.size()[2], x.size()[3])
       
       
        x = self.layer3(x)
        spatial_att3 = self.transformer_3(x.permute(0, 2, 3, 1).view(x.size()[0], x.size()[2]*x.size()[3], 256))
        x = x + spatial_att3.transpose(2, 1).view(x.size()[0], 256, x.size()[2], x.size()[3])
       
        x = self.layer4(x)
        spatial_att4 = self.transformer_4(x.permute(0, 2, 3, 1).view(x.size()[0],  x.size()[2]*x.size()[3], 512))
        x = x + spatial_att4.transpose(2, 1).view(x.size()[0], 512, x.size()[2], x.size()[3])
                
        
        return x


def remove_fc(state_dict):
    """ Remove the fc layer parameter from state_dict. """
    return {key: value for key, value in state_dict.items() if not key.startswith('fc.')}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    '''
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(remove_fc(state_dict))
    '''
    if pretrained:
        #pdb.set_trace()
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        #pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet18_energy_vit(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
                   
def Energy_Function_Max(feature_map):

    with torch.autograd.set_detect_anomaly(True):
        num = feature_map.size()[0]
        channel = feature_map.size()[1]
        spatial_width = feature_map.size()[2]
        spatial_height = feature_map.size()[3]
        device = feature_map.device       
        cov = Covpool.apply(feature_map).to(device)#batch channel channel
        lamda = torch.eye(channel).to(device)
        lamda = torch.where(lamda>0 ,torch.full_like(lamda, 0.01) ,torch.full_like(lamda, 0))
        cov = cov + lamda
                
        u = feature_map.view(num, channel, spatial_width * spatial_height).mean(2).view(num, channel, 1).to(device)#num, 256, 1
        left = (feature_map.view(num, channel, spatial_width * spatial_height) - u).permute(0, 2, 1).to(device)#num,spatial, channel
        index = torch.range(0, spatial_width*spatial_height-1).long().cuda().view(1, spatial_width*spatial_height, 1).repeat(num, 1, 1).to(device)
       
        energy = torch.gather(torch.bmm(torch.bmm(left.clone(), torch.inverse(cov.clone())), left.permute(0, 2, 1).clone()), 2, index).view(num, spatial_width, spatial_height).to(device)
        
        max_value = torch.max(energy.clone().view(num, spatial_width * spatial_height), 1)[0].unsqueeze(1).unsqueeze(1).to(device)
        min_value = torch.min(energy.clone().view(num, spatial_width * spatial_height), 1)[0].unsqueeze(1).unsqueeze(1).to(device)
        #pdb.set_trace()
        energy = (energy.clone() - min_value) / (max_value - min_value)        
        feature_map =  energy.unsqueeze(1) * feature_map + feature_map
        return feature_map


