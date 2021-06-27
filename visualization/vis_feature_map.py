# -*- coding: utf-8 -*-
import numpy as np
import pdb
from collections import defaultdict
import torch.nn.functional as F
import PIL.Image
from PIL import Image
import visualization.utils as utils, cv2
from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tools.nmf import NMF
from tools.utils_nmf import imresize, show_heatmaps
import torch.nn as nn
bb =nn.ReLU()
def stn(x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode='border')
        return x
        
def show_cam_on_image(img, mask, index,imgname):
    #pdb.set_trace()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (192, 256))
    #print(img.shape, heatmap.shape)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)

    cam = cam / np.max(cam)
	
    cv2.imwrite("cam/"+str(index)+'/'+imgname, np.uint8(255 * cam))      
    
def att_show(imgname, output, index):
   
    img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname), 1)
    img = cv2.resize(img, (192, 256))
    img = np.float32(img) / 255
    show_cam_on_image(img, output.cpu().detach().numpy(), index , imgname)    
        
def nmf_show(imgname, output_4):
    for index in range (1,2):
        #flat_features = output_4.permute(0, 2, 3, 1).contiguous().view((-1, output_4.size(1)))
        for i in range(16):
            img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname[i]), 1)
            img_0 = cv2.resize(img, (48, 64))
            #img_2 = cv2.resize(img, (32, 24))
        
            #pdb.set_trace()
            cv2.imwrite("cam/output_0_"+imgname[i], img_0)           
            pdb.set_trace()
            #W, _ = NMF(flat_features, 4, random_seed=0, cuda=True, max_iter=50)
            #heatmaps = W.cpu().view(output_4.size(0), output_4.size(2), output_4.size(3), 4).permute(0,3,1,2) # (N*H*W)xK -> NxKxHxW
            #heatmaps = torch.nn.functional.interpolate(heatmaps, size=(256, 192), mode='bilinear', align_corners=False) ## 14x14 -> 224x224
            #heatmaps /= heatmaps.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0] # normalize by factor (i.e., 1 of K)
            #heatmaps = heatmaps.squeeze().cpu().numpy()
            out_0 = cv2.resize(np.uint8(255*output_4[i][0].cpu().detach().numpy()), (192, 256))
            out_1 = cv2.resize(np.uint8(255*output_4[i][1].cpu().detach().numpy()), (192, 256))
            out_2 = cv2.resize(np.uint8(255*output_4[i][2].cpu().detach().numpy()), (192, 256))
            out_3 = cv2.resize(np.uint8(255*output_4[i][3].cpu().detach().numpy()), (192, 256))
            out_4 = cv2.resize(np.uint8(255*output_4[i][4].cpu().detach().numpy()), (192, 256))          
            
            cv2.imwrite('cam/0'+imgname[i], out_0)
            cv2.imwrite('cam/1'+imgname[i], out_1)
            cv2.imwrite('cam/2'+imgname[i], out_2)
            cv2.imwrite('cam/3'+imgname[i], out_3)
            cv2.imwrite('cam/4'+imgname[i], out_4)
           
                   
def affine(imgname, theta):
    for index in range(len(imgname)):
        pdb.set_trace()
        img = Image.open('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname[index]))
        #img_0 = cv2.resize(img, (48, 64))
        transform = T.Compose([
            T.Resize((256, 192)),       
            T.ToTensor()])
        img = transform(img)  
        output = stn(img.unsqueeze(0), theta[index].unsqueeze(0).cpu().detach())
        output = output.squeeze().permute(1,2,0)
        
        img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname[index]))
        img = cv2.resize(img, (192, 256))
        cv2.imwrite("cam/stn_"+imgname[index], np.uint8(255*(output.numpy())))      
        cv2.imwrite("cam/"+imgname[index], img)          
def show_filter(name, gt_label, feature_map ):
    for i in range(100):
        att = feature_map[i][0]
        #att = (output_0[index]- torch.min(output_0[index])) / (torch.max(output_0[index])- torch.min(output_0[index]))
        att = att.squeeze().cpu().detach().numpy()
        att = cv2.applyColorMap( np.uint8(255*(att)), cv2.COLORMAP_JET)
        #att = att.squeeze().cpu().detach().numpy()
        pdb.set_trace()
        att = cv2.resize(att, (48, 64)) 
        cv2.imwrite("mask/"+str(name)+'/'+str(i)+'.jpg', att)       


def returnCAM(feature_conv, name, weight_softmax):
    
    cam = weight_softmax.dot(feature_conv.reshape((feature_conv.size()[0], 48)))
    cam = cam.reshape(8, 6)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.applyColorMap( np.uint8(255*(cam_img)), cv2.COLORMAP_JET)
    output_cam = cv2.resize(cam_img, (192, 256))
    cv2.imwrite("mask/"+str(attname)+'_'+imgname[index], att)
    
def show_att(imgname,output_1,output_0):
    #output_0 = F.softmax(output_0.view(output_0.size()[0], -1), dim=1)
    #output_0 = output_0.view(output_0.size()[0], 64,48)
    for index in range(len(imgname)):
        #pdb.set_trace()
        for attname in range(0,7):
            
            img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname[index]))
            img = cv2.resize(img, (192, 256))  
            cv2.imwrite("mask/"+imgname[index], img)  
            #for i in range(64):
            att = output_0[index][attname]
            #att = (output_0[index]- torch.min(output_0[index])) / (torch.max(output_0[index])- torch.min(output_0[index]))
            att = att.squeeze().cpu().detach().numpy()
            att = cv2.applyColorMap( np.uint8(255*(att)), cv2.COLORMAP_JET)
            #att = att.squeeze().cpu().detach().numpy()
            #pdb.set_trace()
            att = cv2.resize(att, (192, 256)) 
            cv2.imwrite("mask/"+str(attname)+'_'+imgname[index], att)         
            #cv2.imwrite("mask/"+'_'+imgname[index], np.uint8(255*(att)))
        
def show_mask(imgname,output_0,output_1, output_2,output_3):
   
    for index in range(len(imgname)):
        pdb.set_trace()
        #img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname[index]))
        #img = cv2.resize(img, (192, 256))  
        #cv2.imwrite("cam/"+imgname[index], img)  
        #for i in range(64):
        
        #att = (output_0[index]- torch.min(output_0[index])) / (torch.max(output_0[index])- torch.min(output_0[index]))
        att = output_0[index].squeeze().cpu().detach().numpy()
       
        att = cv2.resize(att, (96, 128))  
        att = cv2.applyColorMap( np.uint8(255*(att)), cv2.COLORMAP_JET)
        cv2.imwrite("mask/"+'_0_'+imgname[index], att)   
        
        att = output_1[index].squeeze().cpu().detach().numpy()
       
        att = cv2.resize(att, (96, 128))  
        att = cv2.applyColorMap( np.uint8(255*(att)), cv2.COLORMAP_JET)
        cv2.imwrite("mask/"+'_1_'+imgname[index], att) 
        
        '''
        att = output_2[index].squeeze().cpu().detach().numpy()
       
        att = cv2.resize(att, (96, 128)) 
        att = cv2.applyColorMap( np.uint8(255*(att)), cv2.COLORMAP_JET)
        cv2.imwrite("mask/"+'_2_'+imgname[index], att) 
        
        att = output_3[index].squeeze().cpu().detach().numpy()
        
        att = cv2.resize(att, (96, 128))
        att = cv2.applyColorMap( np.uint8(255*(att)), cv2.COLORMAP_JET)        
        cv2.imwrite("mask/"+'_3_'+imgname[index], att)         
        '''
        
def vif(imgname,output_depth_0,output_depth_1):
#def vif(imgname, output_rgb_0, output_rgb_1, output_rgb_2):

    for index in range(len(imgname)):
        #pdb.set_trace()
        if True:
            pdb.set_trace()
            #np.save('feature_map/'+imgname[index].split('.')[0]+'.npy', output_depth_0[index].cpu().detach().numpy())
            utils.vis_conv(output_depth_0[index].cpu().detach().numpy(), 5, 7, "conv", str(imgname[index])+"_feature")
            #img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname[index]))
            #img = cv2.resize(img, (192, 256)) 
            #cv2.imwrite("feature_map/"+imgname[index], img) 
        #utils.vis_conv(output_depth_1[index].cpu().detach().numpy(), 8, 16, "conv", str(imgname[index])+"_after_0")
        #utils.vis_conv(output_depth_2[index].cpu().detach().numpy(), 8, 8, "conv", str(imgname[index])+"_before_1")
        #utils.vis_conv(output_depth_3[index].cpu().detach().numpy(), 8, 8, "conv", str(imgname[index])+"_after_1")
        #utils.vis_conv(output_depth_4[index].cpu().detach().numpy(), 8, 8, "conv", str(imgname[index])+"_before_2")
        #utils.vis_conv(output_depth_5[index].cpu().detach().numpy(), 8, 8, "conv", str(imgname[index])+"_after_2")
        #utils.vis_conv(output_fusion_1[index].cpu().detach().numpy(), 8, 8, "conv", str(imgname[index])+"_output_fusion_1")

def show_on_image(imgname, output_0, num_attrbiute, gt_label):
    #pdb.set_trace()
    for index in range(len(imgname)):
        #pdb.set_trace()
        result = defaultdict(list)
        img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/RAP/RAP_dataset/'+str(imgname[index]), 1)
        img_0 = cv2.resize(img, (192, 256))
        for index_attr in [2,3,6,9,10,12,14,15,16,19,21,22,23,26,27,28]:
            if gt_label[index][index_attr] == 1:
                #pdb.set_trace()
                result['name'] = imgname[index]
                loc = np.where(output_0[index_attr][index] == np.max(output_0[index_attr][index]))
                x = loc[0][0]
                y = loc[1][0]
                result['loc'].append([x,y])
                result['cam'].append(output_0[index_attr][index])
                #pdb.set_trace()
                '''
                heatmap = cv2.applyColorMap(np.uint8(255 * (output_0[index_attr][index])), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                #pdb.set_trace()
                cam = heatmap + (np.float32(img_0)/255)
                cam = cam / np.max(cam)
                cam = cv2.resize(cam, (192, 256))
                cv2.imwrite("cam/"+str(index_attr)+"_"+imgname[index], np.uint8(255 * cam))
                '''
                
        np.save('RAP_mask/'+imgname[index].split('.')[0]+'.npy', result)