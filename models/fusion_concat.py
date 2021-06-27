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
from models import MPNCOV
from models.High_Order import *
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import Adam
__all__ = ['fusion_concat']

def softmax(X):
 
    X_exp = X.exp()
 
    partition = X_exp.sum(dim=0, keepdim=True)
 
    return torch.nn.Parameter(X_exp / partition)
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
    
def conv_bn_relu(inp, out,kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size, stride, padding),
        nn.BatchNorm2d(out),
        nn.ReLU()
    )    
# Attention Refinement Module (ARM)
class ARM(nn.Module):
    def __init__(self, in_channels):
        super(ARM, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        # self.bn = nn.BatchNorm1d(in_channels)
        self.sigmod = nn.Sigmoid()
        

    def forward(self, input):
        x = self.global_pool(input)

        x = self.conv_1x1(x)
        # x = self.bn(x)

        x = self.sigmod(x)
        #x = self.up(x)

        out = torch.mul(input, x)
        return out

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFM, self).__init__()
        self.conv_bn_relu = conv_bn_relu(in_channels, out_channels)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_1x1_1 = nn.Conv2d(out_channels,out_channels, 1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv_1x1_2 = nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()
      


    def forward(self, sp, cx):
        #input = torch.cat((sp, cx), 1)
        input = (sp+cx)/2
        feather = self.conv_bn_relu(input)

        x = self.global_pool(feather)

        x = self.conv_1x1_1(x)
        x = self.relu(x)
        x = self.conv_1x1_2(x)
        x = self.sigmod(x)
       

        out = torch.mul(feather, x)
        out = feather + out

        return out

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
        
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, stride=1)
        self.bn2 = norm_layer(2)        
        self.relu = nn.ReLU()
        
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
         
              
        self.maxpool_1 = nn.AdaptiveMaxPool2d((8,6))
        self.channel_scale_1 = nn.Conv2d(64, 512, kernel_size=3,stride=1, padding=1)
        self.norm_1 = nn.LayerNorm(512, eps=1e-6)
        #self.att_1 = SOCA(512)
        #self.arm_1 = ARM(512)
        
        self.maxpool_2 = nn.AdaptiveMaxPool2d((8,6))
        self.channel_scale_2 = nn.Conv2d(128, 512, kernel_size=3,stride=1, padding=1)
        self.norm_2 = nn.LayerNorm(512, eps=1e-6)
        #self.arm_2 = ARM(512)
        
        '''
        self.patch_embedding_0 = nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4))
        self.transformer_0 = Transformer(num_layers=1, dim=64, h=16, w=12, num_heads=1, 
                                      ff_dim=128, dropout=0.1)
        self.channel_scale_0 = nn.Conv2d(64, 256, kernel_size=1,stride=1)                             
                                      
        self.patch_embedding_1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        self.transformer_1 = Transformer(num_layers=1, dim=64, h=32, w=24, num_heads=1, 
                                      ff_dim=128, dropout=0.1)                                      
        self.channel_scale_1 = nn.Conv2d(64, 128, kernel_size=1,stride=1)                                         
        '''
        
        
        #self.att_2 = SOCA(512)
        self.maxpool_3 = nn.AdaptiveMaxPool2d((8,6))
        self.channel_scale_3 = nn.Conv2d(256, 512, kernel_size=3,stride=1, padding=1)
        self.norm_3 = nn.LayerNorm(512, eps=1e-6)
        #self.att_3 = SOCA(512)
        #self.arm_3 = ARM(512)
        
        
        self.norm_4 = nn.LayerNorm(512, eps=1e-6)     
        #self.att_4 = SOCA(512)
        #self.arm_4 = ARM(512)
        #self.para = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        #self.selective_0 = torch.nn.Parameter(torch.Tensor(4,512,1,1).fill_(0.5))
        #self.selective_0 = torch.nn.Parameter(torch.Tensor([0.3, 0.5, 0.7, 1.0])).cuda()
        #self.selective_0.data.copy_(torch.Tensor([0.3,0.5,0.7,1.0]))
        #nn.init.constant_(self.selective_0, 0.5)
        self.high_attention=nn.Sequential(
                        nn.AdaptiveMaxPool2d(1),
                        nn.Conv2d(2048, 128, kernel_size=1,padding=0),
                        nn.ReLU(),
                        nn.Conv2d(128, 2048, kernel_size=1,padding=0),
                        nn.Sigmoid())        
        #self.ffm =FFM(512, 512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        


        
        output_0 = self.maxpool(output_0)

        '''
        branch_0 = self.patch_embedding_0(output_0)#16,12
        branch_0 = self.transformer_0( branch_0.permute(0,2,3,1).view( branch_0.size()[0],  branch_0.size()[2]* branch_0.size()[3], 64))              
        branch_0 = branch_0.permute(0,2,1).view(branch_0.size()[0], 64, 16, 12)
        branch_0 = self.channel_scale_0(branch_0)


        branch_1 = self.patch_embedding_1(output_0)#32,24
        branch_1 = self.transformer_1( branch_1.permute(0,2,3,1).view( branch_1.size()[0],  branch_1.size()[2]* branch_1.size()[3], 64))              
        branch_1 = branch_1.permute(0,2,1).view(branch_1.size()[0], 64, 32, 24)
        branch_1=  self.channel_scale_1(branch_1)
        
        '''
        output_1 = self.layer1(output_0)  
        
        output_2 = self.layer2(output_1)
        
        output_3 = self.layer3(output_2)
        
        output_4 = self.layer4(output_3)
         
        #fusion_concat
        
        
        subset_1 = norm(self.channel_scale_1(self.maxpool_1(output_1)), dim=1)
        subset_2 = norm(self.channel_scale_2(self.maxpool_2(output_2)), dim=1)
        subset_3 = norm(self.channel_scale_3(self.maxpool_3(output_3)), dim=1) 
        subset_4 = norm(output_4, dim=1)
        
        #pdb.set_trace()
        #self.selective_0 = torch.nn.Parameter(self.relu(self.selective_0))
        #self.selective_0 = softmax(self.selective_0)
        #selective_0 = self.relu(self.selective_0)
        #selective_0 /= (torch.sum(selective_0, dim=0) + 0.0001)
    
        output = torch.cat((subset_1, subset_2, subset_3, subset_4), dim=1)
        channel_attention = self.high_attention(output)
        output = subset_1*channel_attention[:,0:512,:,:] + subset_2*channel_attention[:,512:1024,:,:]+ subset_3*channel_attention[:,1024:1536,:,:]+subset_4*channel_attention[:,1536:2048,:,:]
        #output = output+output*channel_attention
        #sum = (self.arm_1(subset_1) + self.arm_2(subset_2) + self.arm_3(subset_3))/3
        #sum =  self.arm_2(subset_2) 
        #sum = self.ffm(subset_4, sum)        
        #sum = self.selective_0[0]*(subset_1) + self.selective_0[1]*(subset_2) + self.selective_0[2]*(subset_3) + self.selective_0[3]*(subset_4)        
        #sum = selective_0[0].unsqueeze(0)*(subset_1) + selective_0[1].unsqueeze(0)*(subset_2) + selective_0[2].unsqueeze(0)*(subset_3) + selective_0[3].unsqueeze(0)*(subset_4)        
        #sum =  ((subset_1) + (subset_2)+(subset_3)+(subset_4))/4              
        return output, output, output#x_cov.view(x_cov.size()[0], 1, 128,96), output_4
        
        #return output_4, output_4, output_4

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

def fusion_concat(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



