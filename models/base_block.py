import math
import pdb
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

'''
class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(512, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def fresh_params(self):
        return self.parameters()

    def forward(self, feature):
        feat = self.avg_pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)
        return x


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)

'''
class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        

        self.bn = nn.BatchNorm1d(nattr)
     
        
        self.fc_1 = nn.Linear(512, nattr)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
   
        
    def fresh_params(self):
        return self.parameters()

    def forward(self, input_feature):

        #feat = self.avg_pool(input_feature).view(input_feature.size(0), -1)
        #feat_1 = self.avg_pool_1(input_feature).view(input_feature.size(0), -1)
        #feat  = torch.cat((feat, feat_1), dim=1)
        #feat = self.avg_pool(input_feature).view(input_feature.size(0), -1)
        feat = self.avg_pool(input_feature).view(input_feature.size(0), -1)
        output_1 = self.fc_1(feat)
        output_1 = self.bn(output_1) 
        #output_1 = self.logits(feat)
        
        
        return output_1
class FeatClassifier(nn.Module):

    #def __init__(self, backbone, classifier, classifier_depth):
    def __init__(self, backbone, classifier):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        #self.classifier_depth = classifier_depth

    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params
        
    def stn_params(self):
        stn = []
        stn.extend([*self.backbone.gausion_att.parameters()])
        #stn.extend([*self.backbone.STN_1.parameters()])
        #stn.extend([*self.backbone.STN_2.parameters()])        
        return stn  
        '''
        depth_network = []
        depth_network.extend([*self.backbone.STN.parameters()])
        depth_network.extend([*self.backbone.bn1_d.parameters()])
        depth_network.extend([*self.backbone.relu_d.parameters()])
        depth_network.extend([*self.backbone.spatial_0.parameters()])
        depth_network.extend([*self.backbone.maxpool_d.parameters()])

        return depth_network                
        '''
   
    def finetune_params(self):
        
        return self.backbone.parameters()
        '''        
        rgb_network = []
        rgb_network.extend([*self.backbone.conv1.parameters()])
        rgb_network.extend([*self.backbone.bn1.parameters()])
        rgb_network.extend([*self.backbone.relu.parameters()])
        rgb_network.extend([*self.backbone.maxpool.parameters()])
        
        rgb_network.extend([*self.backbone.layer1.parameters()])
        rgb_network.extend([*self.backbone.layer2.parameters()])
        rgb_network.extend([*self.backbone.layer3.parameters()])
        rgb_network.extend([*self.backbone.layer4.parameters()])
        
        #rgb_network.extend([*self.backbone.att_rgb.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer1.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer2.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer3.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer4.parameters()])
        
        return rgb_network       
        '''
        
    #def forward(self, x, depth, label=None):
    def forward(self, x, label=None):
        #feat_map, output_depth_0,output_depth_1,output_depth_3, output_rgb_0, output_rgb_1, output_fusion_1 = self.backbone(x, depth)
        #feat_map, theta = self.backbone(x)
        feat_map = self.backbone(x)
        logits= self.classifier(feat_map)
        
        #logits_depth = self.classifier_depth(depth)
        #return logits,output_depth_0,output_depth_1,output_depth_3, output_rgb_0, output_rgb_1, output_fusion_1
        return logits#, theta
