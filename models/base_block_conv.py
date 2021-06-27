import math
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from models.MMA import *
from visualization.vis_feature_map import vif, affine, show_att, show_mask, show_filter
'''
index = []
for i in range(35):
    index.append(torch.from_numpy(np.load(str(i)+'.npy')).cuda().float())
'''
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        
        '''
        self.bn_1 = nn.BatchNorm1d(35)
        self.bn_2 = nn.BatchNorm1d(35)
        self.bn_3 = nn.BatchNorm1d(35)
        self.bn_4 = nn.BatchNorm1d(35)

       
        
        self.conv_1 = conv1x1(64, 35)
        self.conv_2 = conv1x1(128, 35)        
        self.conv_3 = conv1x1(256, 35)
        self.conv_4 = conv1x1(512, 35)
        '''
        self.fc_1 = nn.Linear(512, 35)
        #self.fc_2 = nn.Linear(105, 35)
        self.weight = nn.Parameter(torch.FloatTensor(35, 512))
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.avg_pool_0 = nn.AdaptiveAvgPool2d((8,6))
        #self.avg_pool_1 = nn.AdaptiveAvgPool2d((3,2))
        #self.avg_pool_2 = nn.AdaptiveAvgPool2d((2,2))
        self.nattr = nattr
        for i in range(35):#1408
            setattr(self, 'classifier' + str(i).zfill(2), nn.Sequential(nn.Linear(1408, 1), nn.BatchNorm1d(1)))
            #setattr(self, 'classifier' + str(i).zfill(2), (nn.Parameter(torch.FloatTensor(1, 1408).cuda()), nn.BatchNorm1d(1).cuda()))
            #self.logits_corre[0].weight.data.copy_(torch.tensor(fc_init, dtype=torch.float))   
            #nn.init.xavier_uniform_(getattr(self, 'classifier' + str(i).zfill(2))[0])
    def fresh_params(self):
        return self.parameters()

    def forward(self, input_feature_1, input_feature_2, input_feature_3, input_feature_4, input_feature_5, img_mask):
        '''
        output = self.max_pool(input_feature_1)#+input_feature_2
        output = output.view(input_feature_1.size()[0], -1)
        output = F.normalize(output, p=2, dim=1, eps=1e-12)
        weight = F.normalize(self.weight, p=2, dim=1,eps=1e-12)
        #pdb.set_trace()
        #output = self.fc_1(output)
        output = F.linear(output, weight)
       
        output = self.bn_2(output)
        loss = get_mma_loss(self.weight)
        #conv+bn+max
        '''
        '''
        out_4 = self.conv_4(input_feature_3)#BS , A ,H , W
        out_3 = self.conv_3(input_feature_2)
        out_2 = self.conv_2(input_feature_1)
        
        re_3  = self.bn_3(self.max_pool(out_3).view(-1, 35))
        re_4  = self.bn_4(self.max_pool(out_4).view(-1, 35))
        re_2  = self.bn_2(self.max_pool(out_2).view(-1, 35))
        '''
        
        #pdb.set_trace()
        feat = self.max_pool(input_feature_1).squeeze() #bs,512
        #feat_2 = self.max_pool(input_feature_2).squeeze() #bs,512
        #feat_3 = self.max_pool(input_feature_3).squeeze() #bs,512
        #feat_4 = self.max_pool(input_feature_4).squeeze() #bs,512
        feature_map = []
        weight_loss = []
        #temp = self.conv_4(input_feature_1)
        for i in range(35):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            
            
            #loss = loss + get_mma_loss(self.weight)
            #mask = F.interpolate(img_mask[:,i,:,:].unsqueeze(1), size=[int(8), int(6)], mode="bilinear")
            #feat = self.conv_4(input_feature_1*mask)
            #feat_temp = torch.cat([input_feature_3[i], feat], dim=1)
            #pdb.set_trace()
            feat_temp = torch.cat([input_feature_3[i], input_feature_4[i], input_feature_5[i], feat], dim=1)
            
            #feat = self.avg_pool(input_feature_2[:,:,i,:].unsqueeze(2)).view(input_feature_1.size(0), -1)
            feat_temp = F.normalize(feat_temp, p=2, dim=1, eps=1e-12)
            weight =  F.normalize(classifier[0].weight, p=2, dim=1, eps=1e-12)
            bias = classifier[0].bias
            weight = classifier[0].data
            feat_temp = F.linear(feat_temp, weight)+bias
            #feature_map.append(feat_temp)
            feature_map.append(classifier[1](feat_temp))
            weight_loss.append(weight)
            #feature_map.append(feat)
        #pdb.set_trace()
        loss = get_mma_loss(torch.cat(weight_loss, dim=0))
            
        '''
        feature_map = []
        for i in range(35):
            temp =input_feature*self.fc_1.weight[i].unsqueeze(0).unsqueeze(2).unsqueeze(3) #BS, C, W,H
            temp = torch.mean(temp, dim=1).unsqueeze(1)
            #pdb.set_trace()
            feature_map.append(temp)

        for i in range(35):
            show_filter(i, input_feature[:,index[i],:,:])
       
        pdb.set_trace()
        '''
        
        '''
        feat = self.avg_pool(self.conv_1(input_feature_1)).squeeze()
        output_1 = self.bn_1(feat)
        
        feat = self.avg_pool(self.conv_2(input_feature_2)).squeeze()
        output_2 = self.bn_2(feat)

        feat = self.avg_pool(self.conv_3(input_feature_3)).squeeze()
        output_3 = feat

        feat = self.avg_pool(self.conv_4(input_feature_4)).squeeze()
        output_4 = feat        
        '''
        
        
        #pdb.set_trace()
        '''
        output_1 = []
        for i in range(0, 35):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            #classifier_2 = getattr(self, 'classifier_branch' + str(i).zfill(2))
            #pdb.set_trace()
            #feat = (self.avg_pool(input_feature)*(index[i].unsqueeze(0).unsqueeze(2).unsqueeze(3)) + self.avg_pool(input_feature)).view(input_feature.size(0), -1)
            #feat = torch.matmul(feat, self.fc_1.weight[i])+self.fc_1.bias[i]
            #feat = self.bn_0(0)[feat]
            #feat = torch.matmul(self.fc_1.weight[i], self.avg_pool(feat).view(input_feature))+self.fc_1.bias[i])
            #feat = mask[:,0,:,:].unsqueeze(1)*input_feature #+input_feature
            #feat = self.avg_pool(input_feature).view(input_feature.size(0), -1)
            feat = self.avg_pool(mask).squeeze()
            feat = self.bn_0(feat)
            #show_filter(i, input_feature[:,index[i],:,:])
            #feat = (self.avg_pool(input_feature[:,index[i],:,:])).view(input_feature.size(0), -1)
            #output_1.append(classifier(feat) + classifier_2(feat2))
            #output_1.append(classifier(feat))
            output_1.append(feat)
        '''   
        '''
        for i in range(0, 5):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            #classifier_2 = getattr(self, 'classifier_branch' + str(i).zfill(2))
            #pdb.set_trace()
            feat = (self.avg_pool(input_feature)).view(input_feature.size(0), -1)
            #feat = torch.matmul(feat, self.fc_1.weight[i])+self.fc_1.bias[i]
            #feat = self.bn_0(0)[feat]
            #feat = torch.matmul(self.fc_1.weight[i], self.avg_pool(feat).view(input_feature))+self.fc_1.bias[i])
            #feat = mask[:,0,:,:].unsqueeze(1)*input_feature #+input_feature
            #feat = (self.avg_pool(feat)+self.avg_pool(input_feature[:,:,0:1,:])).view(feat.size(0), -1)
            #show_filter(i, input_feature[:,index[i],:,:])
            #feat = (self.avg_pool(input_feature[:,index[i],:,:])).view(input_feature.size(0), -1)
            #output_1.append(classifier(feat) + classifier_2(feat2))
            output_1.append(classifier(feat))
        
        for i in range(5, 15):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            #classifier_2 = getattr(self, 'classifier_branch' + str(i).zfill(2))
            #pdb.set_trace()
            feat = (self.avg_pool(input_feature)).view(input_feature.size(0), -1)
            #feat = torch.matmul(feat, self.fc_1.weight[i])+self.fc_1.bias[i]
            #feat = self.bn_0(0)[feat]
            #feat = torch.matmul(self.fc_1.weight[i], self.avg_pool(feat).view(input_feature))+self.fc_1.bias[i])
            #feat = mask[:,0,:,:].unsqueeze(1)*input_feature #+input_feature
            #feat = (self.avg_pool(feat)+self.avg_pool(input_feature[:,:,0:1,:])).view(feat.size(0), -1)
            #show_filter(i, input_feature[:,index[i],:,:])
            #feat = (self.avg_pool(input_feature[:,index[i],:,:])).view(input_feature.size(0), -1)
            #output_1.append(classifier(feat) + classifier_2(feat2))
            output_1.append(feat)           
        for i in range(15, 21):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            #classifier_2 = getattr(self, 'classifier_branch' + str(i).zfill(2))
            #pdb.set_trace()
            feat = (self.avg_pool(input_feature)).view(input_feature.size(0), -1)
            #feat = torch.matmul(feat, self.fc_1.weight[i])+self.fc_1.bias[i]
            #feat = self.bn_0(0)[feat]
            #feat = torch.matmul(self.fc_1.weight[i], self.avg_pool(feat).view(input_feature))+self.fc_1.bias[i])
            #feat = mask[:,0,:,:].unsqueeze(1)*input_feature #+input_feature
            #feat = (self.avg_pool(feat)+self.avg_pool(input_feature[:,:,0:1,:])).view(feat.size(0), -1)
            #show_filter(i, input_feature[:,index[i],:,:])
            #feat = (self.avg_pool(input_feature[:,index[i],:,:])).view(input_feature.size(0), -1)
            #output_1.append(classifier(feat) + classifier_2(feat2))
            output_1.append(feat)            
        for i in range(21, 25):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            #classifier_2 = getattr(self, 'classifier_branch' + str(i).zfill(2))
            #pdb.set_trace()
            feat = (self.avg_pool(input_feature)).view(input_feature.size(0), -1)
            #feat = torch.matmul(feat, self.fc_1.weight[i])+self.fc_1.bias[i]
            #feat = self.bn_0(0)[feat]
            #feat = torch.matmul(self.fc_1.weight[i], self.avg_pool(feat).view(input_feature))+self.fc_1.bias[i])
            #feat = mask[:,0,:,:].unsqueeze(1)*input_feature #+input_feature
            #feat = (self.avg_pool(feat)+self.avg_pool(input_feature[:,:,0:1,:])).view(feat.size(0), -1)
            #show_filter(i, input_feature[:,index[i],:,:])
            #feat = (self.avg_pool(input_feature[:,index[i],:,:])).view(input_feature.size(0), -1)
            #output_1.append(classifier(feat) + classifier_2(feat2))
            output_1.append(feat)               
        for i in range(25, 30):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            #classifier_2 = getattr(self, 'classifier_branch' + str(i).zfill(2))
            #pdb.set_trace()
            feat = (self.avg_pool(input_feature)).view(input_feature.size(0), -1)
            #feat = torch.matmul(feat, self.fc_1.weight[i])+self.fc_1.bias[i]
            #feat = self.bn_0(0)[feat]
            #feat = torch.matmul(self.fc_1.weight[i], self.avg_pool(feat).view(input_feature))+self.fc_1.bias[i])
            #feat = mask[:,0,:,:].unsqueeze(1)*input_feature #+input_feature
            #feat = (self.avg_pool(feat)+self.avg_pool(input_feature[:,:,0:1,:])).view(feat.size(0), -1)
            #show_filter(i, input_feature[:,index[i],:,:])
            #feat = (self.avg_pool(input_feature[:,index[i],:,:])).view(input_feature.size(0), -1)
            #output_1.append(classifier(feat) + classifier_2(feat2))
            output_1.append(feat)      
        for i in range(30, 35):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            #classifier_2 = getattr(self, 'classifier_branch' + str(i).zfill(2))
            #pdb.set_trace()
            feat = (self.avg_pool(input_feature)).view(input_feature.size(0), -1)
            #feat = torch.matmul(feat, self.fc_1.weight[i])+self.fc_1.bias[i]
            #feat = self.bn_0(0)[feat]
            #feat = torch.matmul(self.fc_1.weight[i], self.avg_pool(feat).view(input_feature))+self.fc_1.bias[i])
            #feat = mask[:,0,:,:].unsqueeze(1)*input_feature #+input_feature
            #feat = (self.avg_pool(feat)+self.avg_pool(input_feature[:,:,0:1,:])).view(feat.size(0), -1)
            #show_filter(i, input_feature[:,index[i],:,:])
            #feat = (self.avg_pool(input_feature[:,index[i],:,:])).view(input_feature.size(0), -1)
            #output_1.append(classifier(feat) + classifier_2(feat2))
            output_1.append(feat)      
          
        
        #pdb.set_trace()
        '''
        #output_1 = torch.cat(output_1, dim=1)
        
        feature_map = torch.cat(feature_map, dim=1)
        return feature_map, loss#, feature_map, feature_map#, mask#torch.cat(mask, dim=1)#, output_2#, output_2#, output_3#, output_2#, output_2_inter#, x_2
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
        stn.extend([*self.backbone.fix.parameters()])
     
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
        rgb_network.extend([*self.backbone.up_dim.parameters()])
        rgb_network.extend([*self.backbone.bn_2.parameters()])
        rgb_network.extend([*self.backbone.cbam.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer1.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer2.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer3.parameters()])
        #rgb_network.extend([*self.backbone.att_rgb_layer4.parameters()])
        
        return rgb_network       
        
        '''
    #def forward(self, x, depth, label=None):
    def forward(self, x, img_mask,label=None):
        #feat_map, output_depth_0,output_depth_1,output_depth_3, output_rgb_0, output_rgb_1, output_fusion_1 = self.backbone(x, depth)
        #feat_map, theta = self.backbone(x)
        feat, feat_2, feat_3, feat_4,feat_5, re_2, re_3, re_4, re_5 = self.backbone(x, img_mask,label)
        logits_1, loss = self.classifier(feat, feat_2, feat_3, feat_4, feat_5,img_mask)

        return logits_1, re_2, re_3, re_4, re_5, loss#,feat_map_2, feat_map_3#, theta
