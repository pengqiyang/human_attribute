import time
import torch.nn.functional as F 
import numpy 
import pdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from visualization.vis_feature_map import vif, affine, show_att, show_mask, nmf_show, att_show
from tools.utils import AverageMeter, to_scalar, time_str
from models.MMA import *
from torch.autograd import Variable
from tools.detecotor_utils import *


def dice_loss_with_sigmoid(sigmoid, targets, smooth=1.0):
	"""
	sigmoid: (torch.float32)  shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
	sigmoid, targets = sigmoid.squeeze(), targets.squeeze()
	#pdb.set_trace()
	outputs = torch.squeeze(sigmoid, dim=1)

	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(1,2)) + smooth) / (outputs.sum(dim=(1,2))+targets.sum(dim=(1,2)) + smooth))
	dice = dice.mean()
	return dice
def generate_flip_grid(w, h):
	# used to flip attention maps
	x_ = torch.arange(w).view(1, -1).expand(h, -1)
	y_ = torch.arange(h).view(-1, 1).expand(-1, w)
	grid = torch.stack([x_, y_], dim=0).float().cuda()
	grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
	grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
	grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1

	grid[:, 0, :, :] = -grid[:, 0, :, :]
	return grid
grid_l = generate_flip_grid(48, 64)
grid_s = generate_flip_grid(24, 32)
def l2_norm(input, axit=1):
    norm = torch.norm(input,2,axit,True)
    output = torch.div(input, norm)
    return output
def batch_trainer(epoch, model, train_loader, criterion, optimizer, loss):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[0]['lr']

    for step, (img_l, img_s, img_lf, img_sf, gt_label,img_mask, imgname) in enumerate(train_loader):
        #print(step)
       
        batch_time = time.time()
        
        img_l, img_s, img_lf, img_sf, gt_label, img_mask = img_l.cuda(), img_s.cuda(), img_lf.cuda(), img_sf.cuda(), gt_label.cuda(), img_mask.cuda()
        #pdb.set_trace()
        train_logit_l, train_logit_2, train_logit_3, train_logit_4, train_logit_5,  _= model(img_l, img_mask, gt_label)
        
        #get_att(imgname, gt_label, train_logit_4, train_logit_4)
        #pdb.set_trace()
        #train_logit_s, cha_att_s, spa_att_s = model(img_s, gt_label)
        #train_logit_lf, cha_att_lf, spa_att_lf = model(img_lf, gt_label)
        #train_logit_sf, cha_att_sf, spa_att_sf = model(img_sf, gt_label)
        '''
        show_mask(imgname, spa_att_l, spa_att_lf, spa_att_s, spa_att_sf)
        pdb.set_trace()
        '''
        #flip
        '''
        flip_grid_large = grid_l.expand(img_l.size()[0], -1, -1, -1)
        flip_grid_large = Variable(flip_grid_large, requires_grad = False)
        flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
        hm_l = F.grid_sample(spa_att_l, flip_grid_large, mode = 'bilinear', padding_mode = 'border')
        '''
        #pdb.set_trace()

        '''
        flip_grid_small = grid_s.expand(img_l.size()[0], -1, -1, -1)
        flip_grid_small = Variable(flip_grid_small, requires_grad = False)
        flip_grid_small = flip_grid_small.permute(0, 2, 3, 1)
        hm_s = F.grid_sample(spa_att_s, flip_grid_small, mode = 'bilinear',padding_mode = 'border')
        '''
        
        #flip_loss =  F.mse_loss(hm_l, spa_att_lf)  #+ F.mse_loss(hm_s, spa_att_sf)
        
        #scale
        
        '''
        hm_l = F.upsample(spa_att_l, (32, 24))
        #hm_s = F.upsample(spa_att_lf, (32, 24))
        scale_loss = F.mse_loss(hm_l, spa_att_s) #+ F.mse_loss(hm_s, spa_att_sf)            
        '''
        
       
        if loss  == 'CONSIST_LOSS':
            train_loss_l = 0.3*criterion(train_logit_l, gt_label) + 0.3*criterion(train_logit_3, gt_label)+ 0.7*criterion(train_logit_4, gt_label) +criterion(train_logit_5, gt_label)#+ 0.7*criterion(train_logit_3, gt_label)+1*criterion(train_logit_4, gt_label)#+train_logit_4#+criterion(train_logit_2, gt_label)
            '''
            #每个属性关注的区域尽量正交
          
            logit = train_logit_3.permute(1,0,2,3)#35, BS, H, W
            logit = logit.reshape(64, 35 ,-1)
            
            #pdb.set_trace()
            dot = torch.matmul(logit, logit.permute(0,2,1))#BS , 35, 35
            mask = torch.from_numpy(np.load('mask.npy', allow_pickle=True)).float().cuda()
            mask = mask.unsqueeze(0)
            mask = mask.expand_as(dot)
            dot = dot[mask==1]
            '''
            '''
            for i in range(64):
                #pdb.set_trace()
                dot[i] = dot[i] -  torch.diag(torch.diag(dot[i]))
            #pdb.set_trace()
            '''
            '''
            loss_1 = torch.mean(dot)
            
            #每个属性关注的距离关系
            loc = 0
            for i in range(64):
                #pdb.set_trace()
                feat = logit[i].reshape(35, -1)
                sort, index = torch.max(feat, dim=1)
                index  = (index/12).float()
                height_0 = torch.mean(index[0:5])
                height_1 = torch.mean(index[5:15])
                height_2 = torch.mean(index[15:21])
                height_3 = torch.mean(index[21:25])
                height_4 = torch.mean(index[25:30])                
                loc = loc+torch.mean(torch.exp(-(height_3-height_4))+torch.exp(-(height_3-height_2))+torch.exp(-(height_2-height_1))+torch.exp(-(height_4-height_2))+torch.exp(-(height_1-height_0)))
            
            
            
            '''
            #pdb.set_trace()
            '''
            for name, m in model.named_modules():
            # 'name' can be used to exclude some specified layers
            
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    #pdb.set_trace()
                    mma_loss = get_mma_loss(m.weight)
                    train_loss_l = train_loss_l + 0.03 * mma_loss
            '''
            #loc = get_feat_loss(feature_map, gt_label)
            #
            #train_loss_s = criterion(train_logit_s, gt_label)
            #train_loss_lf = criterion(train_logit_lf, gt_label)
            #train_loss_sf = criterion(train_logit_sf, gt_label)
            #keypoints  = cha_att_l.view(-1,35,2)
            #var_h = torch.std(keypoints[:,:,0], dim=1)
            #var_w = torch.std(keypoints[:,:,1], dim=1)
            #height_0, height_1, height_2, height_3 = keypoints[:,0], keypoints[:,3],  keypoints[:,6],  keypoints[:,9]
            #loc = torch.sum(torch.exp(-(height_3-height_2))+torch.exp(-(height_2-height_1))+torch.exp(-(height_1-height_0)))
            #height = torch.mean(torch.mean(torch.max(keypoints, dim=2)[1].float(), dim=2), dim=0)
            #height_0, height_1, height_2, height_3 = height[0], height[1],  height[2],  height[3]
            #loc = torch.sum(torch.exp(-(height_3-height_2))+torch.exp(-(height_2-height_1))+torch.exp(-(height_1-height_0)))
            train_loss = train_loss_l+_#+loss_1 + 0.03*loc  #+ 30*(torch.exp(-torch.sum(var_w)))
            #pdb.set_trace()
            #train_loss = train_loss_l #+ dice_loss_with_sigmoid(spa_att_l, img_seg)
        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logit_l)
        #train_probs_2 = torch.sigmoid(train_logit_2)
        #train_probs_3 = torch.sigmoid(train_logit_3)
        #train_probs_4 = torch.sigmoid(train_logit_4)
        #train_max = (train_probs + train_probs_2)/2
        #preds_probs.append(train_max.detach().cpu().numpy())
        preds_probs.append(train_probs.detach().cpu().numpy())
        log_interval = 20
        
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {time.time() - batch_time:.2f}s ',
                  f'train_loss:{loss_meter.val:.4f}')

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs


# @torch.no_grad()
def valid_trainer(model, valid_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    
    
    
    with torch.no_grad():
        for step, (img_l, img_s, img_lf, img_sf, gt_label, img_mask, imgname) in enumerate(tqdm(valid_loader)):
            #pdb.set_trace()
            img_l, img_s, img_lf, img_sf, gt_label, img_mask = img_l.cuda(), img_s.cuda(), img_lf.cuda(), img_sf.cuda(), gt_label.cuda(), img_mask.cuda()
            valid_logit_l, valid_logit_2, valid_logit_3, valid_logit_4, valid_logit_5, att_map = model(img_l, img_mask, gt_label)
            #valid_logit_s, cha_att_s, spa_att_s = model(img_s, gt_label)
            #valid_logit_lf, cha_att_lf, spa_att_lf = model(img_lf, gt_label)
            #valid_logit_sf, cha_att_sf, spa_att_sf = model(img_sf, gt_label)
            #show_mask(imgname, spa_att_l, cha_att_l, spa_att_l, spa_att_l)
            
            #loc = get_feat_loss(feature_map, gt_label)
            #pdb.set_trace()
            gt_list.append(gt_label.cpu().numpy())
            
            gt_label[gt_label == -1] = 0
            #valid_logits, cha_att, spa_att = model(imgs)
            
            #pdb.set_trace()
            

                
            #valid_loss = 0
            valid_loss =  criterion(valid_logit_l,  gt_label)#+criterion(valid_logit_2,  gt_label)#+ 0.2*(torch.sum(torch.abs(cha_att))+torch.sum(torch.abs(spa_att)))
            valid_probs = torch.sigmoid(valid_logit_l)
            #valid_probs_2 = torch.sigmoid(valid_logit_2)
            #valid_probs_3 = torch.sigmoid(valid_logit_3)
            #valid_probs_4 = torch.sigmoid(valid_logit_4)
            #pdb.set_trace()
    
            
            

            
            #pred_max = (valid_probs + valid_probs_2)/2
            #preds_probs.append(pred_max.cpu().numpy())
            preds_probs.append(valid_probs.detach().cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))
            
            #show_filter(imgname, gt_label, valid_logit_2)
            #pdb.set_trace()
            #nmf_show(imgname, feature_map)
            #show_att(imgname, mask,mask )
            #affine(imgname, theta)
            #vif(imgname, valid_logit_4, valid_logit_4)
            #return 0
            #get_mask_block(imgname, gt_label, valid_logit_l, valid_logit_3, valid_logit_2)
            #get_att(imgname, gt_label, valid_logit_2, valid_logit_l)
            #pdb.set_trace()
            #get_detector(gt_label, valid_logit_4,valid_probs_2, valid_logit_3)
    #np.save('part_detector.py', part_detector)
    valid_loss = loss_meter.avg
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    #save_part()
    return valid_loss, gt_label, preds_probs
