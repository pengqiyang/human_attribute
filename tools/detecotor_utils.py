import time
import torch.nn.functional as F 
import numpy 
import pdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.autograd import Variable
import pdb
from visualization.vis_feature_map import *
from collections import defaultdict
part_detector = []
total =[]
det=[]
loc_save = defaultdict(list)
for i in range(35):
    det.append(torch.from_numpy(np.load('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/tools/layer3/'+str(i)+'.npy', allow_pickle=True)).cuda().float())
for i in range(35):
    total.append(torch.zeros(512).cuda())
    part_detector.append([])
sum_ = torch.zeros(35)
#统计最后一层不同属性对每一个filter的激活次数
def activation_channel_tongji(feature_map, gt_label):
    for i in range(gt_label.size()[0]):
        channel = feature_map[i]#C
        sort, indices = torch.sort(channel, descending=True)
        for j in range(35):
            if (gt_label[i][j]==1):
                sum_[j]=sum_[j]+1
                total[j][indices[:400]] =  total[j][indices[:400]] + 1  
                
#得到attribute-specific的filter
def channel_gongtong():
    pdb.set_trace()
    for i in range(35):
        
        total[i] = total[i]/sum_[i]
        sort, indices = torch.sort(total[i], descending=True)
        np.save(str(i)+'.npy', indices.cpu().detach().numpy() )#保存每个通道的激活次数
    pdb.set_trace()
    for i in range(35):
        for j in range(i,35):
            heji = torch.zeros(512)
            heji_ = torch.zeros(512)
            sort, indices = torch.sort(total[i], descending=True)
            sort_, indices_ = torch.sort(total[j], descending=True)
            #pdb.set_trace()
            heji[indices[:100]] =1
            heji_[indices_[:100]] =1
            
            print(str(i) +' '+str(j)+' '+str(torch.sum(heji*heji_)))
    pdb.set_trace()

#valid_logit:B, A, W , H
#feature_map: B C W H
#result:B , A
def get_detector(gt_label, valid_logit, result, feature_map):
    for i in range(gt_label.size()[0]):
        for j in range(35):
            if (gt_label[i][j]==1) and (result[i][j]>=0.5):
                #pdb.set_trace()
                index = torch.where(valid_logit[i][j] ==  torch.max(valid_logit[i][j]))
                x,y=index[0].data, index[1].data                       
                feature = feature_map[i,:,x,y]
                bb  = feature
                #bb = torch.nn.functional.normalize(feature.unsqueeze(2), p=2, dim=0, eps=1e-12, out=None)
                part_detector[j].append(bb.cpu().detach().numpy())
  
def save_part():
    np.save('PETA_layer3_loc.npy', loc_save)
    #np.save('part_detector_layer4.npy', part_detector)
#feature_map: 35, bs, h, w
def get_att(imgname, gt_label, feature_map, valid_logit):
    #pdb.set_trace()
    feature_map = feature_map.permute(1,0,2,3)
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    for i in range(gt_label.size()[0]):
        for j in range(35):
            if valid_logit[i][j]>=0 and gt_label[i][j]==1:
                att_show(imgname[i], feature_map[i][j], j)


#valid_logit:B, A
#feature_map: B A W H
#loc: bs, 35, 2
def get_mask_block(imgname, gt_label, valid_logit, feature_map, loc):
    lineType = 4
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 1 # 可以为 0 、4、8
    mean=[0.4914, 0.4822, 0.4465]
    #pdb.set_trace()
    for i in range(gt_label.size()[0]):
        #aa = torch.nn.functional.normalize(feature_map[i], p=2, dim=0, eps=1e-12, out=None)
        #image = []
        pos_x=[]
        pos_y=[]
        #pos_acc_x=[]
        #pos_acc_y=[]
        #acc_val=[]
        #val = []
        #img = cv2.imread('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA/images/'+str(imgname[i]))
        #img = cv2.resize(img, (196, 256))        
        #img_1, img_2, img_3 = img, img, img
        
        for j in range(35): 
            if gt_label[i][j] == 1:
                
                #index = torch.where(feature_map[i][j] ==  torch.max(feature_map[i][j]))
                #print(index)
                
                x,y=loc[i][j][0].data, loc[i][j][1].data 
                #print(str(x)+' '+str(y))
                pos_x.append(x)
                pos_y.append(y)
                #val.append(torch.max(feature_map[i][j]))
                #if gt_label[i][j]==1 and valid_logit[i][j]>=0.5 :
                '''
                bb = torch.nn.functional.normalize(det[j].unsqueeze(2).unsqueeze(3), p=2, dim=1, eps=1e-12, out=None)
               
                sim_0 = aa*bb[0]
                sim_0 = torch.sum(sim_0, dim=0)
                sim_1 = aa*bb[1]
                sim_1 = torch.sum(sim_1, dim=0)
                sim_2= aa*bb[2]
                sim_2 = torch.sum(sim_2, dim=0)
                sim = torch.max(torch.max(sim_0, sim_1), sim_2)           
                att_show(imgname[i], sim, j)
                #image.append(sim)
                '''
        '''
        index = np.argmax(val[:5])
        pos_acc_x.append(point_x[index])
        pos_acc_y.append(point_y[index])
        acc_val.append(val[index])
        
        index = np.argmax(val[5:15])
        pos_acc_x.append(point_x[index+5])
        pos_acc_y.append(point_y[index+5])
        acc_val.append(val[5+index])
        '''
        '''
        np.argmax(val[7:10])
        pos_acc_x.append(point_x[index+7])
        pos_acc_y.append(point_y[index+7])
        acc_val.append(val[7+index])
        
        pos_acc_x.append(point_x[10])
        pos_acc_y.append(point_y[10])
        acc_val.append(val[10+index])
        
        np.argmax(val[11:15])
        pos_acc_x.append(point_x[index+11])
        pos_acc_y.append(point_y[index+11])
        acc_val.append(val[11+index])
        '''
        '''
        index = np.argmax(val[17:21])
        pos_acc_x.append(point_x[index+17])
        pos_acc_y.append(point_y[index+17])
        acc_val.append(val[17+index])
        
        np.argmax(val[21:25])
        pos_acc_x.append(point_x[21+index])
        pos_acc_y.append(point_y[21+index])
        acc_val.append(val[21+index])
        
        index=np.argmax(val[25:30])
        pos_acc_x.append(point_x[25+index])
        pos_acc_y.append(point_y[25+index])
        acc_val.append(val[25+index])
        
        
        index = np.argmax(val[30:34])
        pos_acc_x.append(point_x[30+index])
        pos_acc_y.append(point_y[30+index])
        acc_val.append(val[30+index])
        '''
             
        
        for k in range(len(pos_x)):
            #print(16*pos_y[k].cpu().detach().numpy()[0])
            #print(16*pos_x[k].cpu().detach().numpy()[0])
            #print(acc_val[k])
            #ptLeftTop = (16*pos_y[k]-2,16*pos_x[k]-2)
            #ptRightBottom = (16*pos_y[k]+2,16*pos_x[k]+2)
            x1 = 16*pos_y[k].cpu().detach().numpy()
            y1 = 16*pos_x[k].cpu().detach().numpy()
            #pdb.set_trace()
            loc_save[imgname[i]].append([x1, y1])
            '''
            #cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            img_1[x1-2:x1+2, y1-2:y1+2, 0] = mean[2]
            img_1[x1-2:x1+2, y1-2:y1+2, 1] = mean[1]
            img_1[x1-2:x1+2, y1-2:y1+2, 2] = mean[0]
            
            img_2[x1-4:x1+4, y1-4:y1+4, 0] = mean[2]
            img_2[x1-4:x1+4, y1-4:y1+4, 1] = mean[1]
            img_2[x1-4:x1+4, y1-4:y1+4, 2] = mean[0]

            img_3[x1-8:x1+8, y1-8:y1+8, 0] = mean[2]
            img_3[x1-8:x1+8, y1-8:y1+8, 1] = mean[1]
            img_3[x1-8:x1+8, y1-8:y1+8, 2] = mean[0] 
            '''
            #cv2.circle(img, (16*pos_y[k].cpu().detach().numpy()[0], 16*pos_x[k].cpu().detach().numpy()[0]), point_size, point_color, thickness)
            #cv2.imwrite('images/'+imgname[i], img)
            #pdb.set_trace()
        #pdb.set_trace()
        #img = cv2.resize(img, (192,256))
        print(imgname[i])
        '''
        cv2.imwrite('images/'+imgname[i].split('.')[0]+'_4.png', img_1)
        cv2.imwrite('images/'+imgname[i].split('.')[0]+'_8.png', img_2)
        cv2.imwrite('images/'+imgname[i].split('.')[0]+'_16.png', img_3)   
        '''        
        #np.save('/media/data1/pengqy/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA_mask/'+str(imgname[i].split('.')[0])+'.npy', image)