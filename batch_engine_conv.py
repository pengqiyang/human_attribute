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
from torch.autograd import Variable


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

    for step, (imgs, gt_label, imgname) in enumerate(train_loader):
        
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        
        train_logit_1, train_logit_2, train_logit_3, train_logit_4 = model(imgs)
        

        if loss  == 'Multi_Level_Loss':
            train_loss = 0.1*criterion(train_logit_1, gt_label) + 0.3*criterion(train_logit_2, gt_label)+ 0.7*criterion(train_logit_3, gt_label) +criterion(train_logit_4, gt_label)
            
        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logit_4)
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
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
           
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            valid_logit_1, valid_logit_2, valid_logit_3, valid_logit_4 = model(imgs)
           
            #pdb.set_trace()
            gt_list.append(gt_label.cpu().numpy())
            
            gt_label[gt_label == -1] = 0
            #valid_logits, cha_att, spa_att = model(imgs)
            
            #pdb.set_trace()
            

                
            #valid_loss = 0
            valid_loss =  criterion(valid_logit_4,  gt_label)
            valid_probs = torch.sigmoid(valid_logit_4)
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
