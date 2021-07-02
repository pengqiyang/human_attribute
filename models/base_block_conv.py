import math
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np

from visualization.vis_feature_map import vif, affine, show_att, show_mask, show_filter

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        
        
        self.bn_1 = nn.BatchNorm1d(nattr)
        self.bn_2 = nn.BatchNorm1d(nattr)
        self.bn_3 = nn.BatchNorm1d(nattr)
        self.bn_4 = nn.BatchNorm1d(nattr)

       
        
        self.conv_1 = conv1x1(64, nattr)
        self.conv_2 = conv1x1(128, nattr)        
        self.conv_3 = conv1x1(256, nattr)
        self.conv_4 = conv1x1(512, nattr)
        
        
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.nattr = nattr
        
    def fresh_params(self):
        return self.parameters()

    def forward(self, input_feature_1, input_feature_2, input_feature_3, input_feature_4):

        out_4 = self.conv_4(input_feature_4)#BS , A ,H , W        
        out_3 = self.conv_3(input_feature_3)#BS , A ,H , W
        out_2 = self.conv_2(input_feature_2)
        out_1 = self.conv_1(input_feature_1)
        
        re_3  = self.bn_3(self.max_pool(out_3).view(-1, self.nattr))
        re_4  = self.bn_4(self.max_pool(out_4).view(-1, self.nattr))
        re_2  = self.bn_2(self.max_pool(out_2).view(-1, self.nattr))
        re_1  = self.bn_1(self.max_pool(out_1).view(-1, self.nattr))        
        
        
        return re_1, re_2, re_3, re_4


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        
    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params
        
    def stn_params(self):
        stn = []
        stn.extend([*self.backbone.fix.parameters()])
      
        return stn  

   
    def finetune_params(self):
        
        return self.backbone.parameters()

   
    def forward(self, x, label=None):

        feat_1, feat_2, feat_3, feat_4  = self.backbone(x)
        logits_1, logits_2, logits_3, logits_4 = self.classifier(feat_1, feat_2, feat_3, feat_4)

        return logits_1, logits_2, logits_3, logits_4