import math
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from .transformer import Transformer
import pdb
import numpy as np
fc_init = torch.from_numpy(np.load('tools/gongxian_norm.npy')).cuda().float()
class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        
        self.logits = nn.Sequential(
            nn.Linear(2048, nattr),
            nn.BatchNorm1d(nattr)
        )
       
        #self.bn = nn.BatchNorm1d(nattr)
        
        #self.logits_corre = nn.Sequential(
        #    nn.Linear(nattr, nattr),
        #    nn.BatchNorm1d(nattr)
        #)  
        
        #self.fc_1 = nn.Linear(512, nattr)       
        self.fc_0 = nn.Linear(2048, nattr)
        self.bn_0 = nn.BatchNorm1d(nattr)
        
        self.fc_1 = nn.Linear(nattr, nattr)
        self.bn_1 = nn.BatchNorm1d(nattr)
        
        self.relu = nn.ReLU()

        #self.transformer_1 = Transformer(num_layers=1, dim=24, num_heads=1, 
        #                              ff_dim=96, dropout=0.1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(2048, nattr, kernel_size=3, stride=1)
        #self.conv2 = nn.Conv2d(512, nattr, kernel_size=3, stride=1)
        #self.sig = nn.Sigmoid()
        self.nattr = nattr
        self.weight = fc_init
        #self.logits_corre[0].weight.data.copy_(torch.tensor(fc_init, dtype=torch.float))   
        
    def fresh_params(self):
        return self.parameters()

    def forward(self, input_feature):
        #conv + bn
        '''
        feature = self.conv1(input_feature)     
        output_1 = self.avg_pool(feature).view(feature.size(0), -1)    
        output_1 = self.bn(output_1)     
        '''
        #version1: conv + fc + bn
        '''
        feature = self.conv1(input_feature)     
        output_1 = feature.view(feature.size(0), -1)
        output_1 = self.logits_corre(output_1)
        '''
        
        #version2: conv + fc + bn
        '''
        feature = self.conv1(input_feature)
        output_1 = self.avg_pool(feature).view(feature.size(0), -1)
        output_1 = self.logits_corre(output_1)
        '''
        
        #fc + bn
        '''
        feat = self.avg_pool(input_feature).view(input_feature.size(0), -1)
        output_2 = self.logits(feat) 
        '''

        '''
        feature = self.conv1(input_feature)     
        output_1 = self.avg_pool(feature).view(feature.size(0), -1)  
        output_1 = self.logits_corre(output_1)      
        '''
        
        feat = self.avg_pool(input_feature).view(input_feature.size(0), -1)
        output_1 = self.fc_0(feat)
        output_1 = self.bn_0(output_1)
        #pdb.set_trace()
        output_2 = torch.matmul(output_1, self.weight)
        
        
        #output_2 = self.fc_1(output_1) 
        output_2 = self.bn_1(output_2) 
        
        '''
        #output_1 = self.logits[0](feat)  
        output_1 = self.logits(feat)      
        #output_2 = self.logits_corre(output_1)
        '''      

        
        #output_1 = self.logits(feat)
        return output_1, output_2#, output_2#, output_3#, output_2#, output_2_inter#, x_2


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
        
    def fresh_depth_params(self):
        params = self.classifier_depth.fresh_params()
        return params
        
    def depth_params(self):
        return self.backbone.depth_network.parameters()   
        '''
        depth_network = []
        depth_network.extend([*self.backbone.conv1_d.parameters()])
        depth_network.extend([*self.backbone.bn1_d.parameters()])
        depth_network.extend([*self.backbone.relu_d.parameters()])
        depth_network.extend([*self.backbone.spatial_0.parameters()])
        depth_network.extend([*self.backbone.maxpool_d.parameters()])

        return depth_network                
        '''
    def finetune_params(self):
    
        return self.backbone.parameters()

    def forward(self, x, label=None):
    #def forward(self, x, label=None):

        feat_map, feat_map_0 , feat_map_1  = self.backbone(x)
        #output_1, output_2= self.classifier(feat_map)
        output_2, output_3 = self.classifier(feat_map)
        return output_2, output_3#, output_2   #return logits, logits_depth
