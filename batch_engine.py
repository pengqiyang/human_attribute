import time
import torch.nn.functional as F 
import pdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from visualization.vis_feature_map import vif, affine, show_att, show_on_image
from tools.utils import AverageMeter, to_scalar, time_str
from models.MMA import *
from visualization.grad_cam import *
'''
mask = torch.zeros(128,96)
for i in range(48, 96):
    for j in range(24, 64):
        mask[i][j] = 1


mask = torch.ones(51)
for i in range(6):
    mask[i] = 0
mask = mask.cuda().float().unsqueeze(0)
'''
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

    lr = optimizer.param_groups[1]['lr']

    for step, (imgs, gt_label, imgname) in enumerate(train_loader):
        #print(step)
       
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
       
        train_logits = model(imgs, gt_label)
        if loss == 'KL_LOSS':
            sim = np.load('src/sim.npy')
            cls_weight = model.state_dict()['module.classifier.logits.0.weight']
            #pdb.set_trace()
            cls_weight_t = torch.transpose(cls_weight, 1, 0)
            cls = torch.mm(cls_weight, cls_weight_t)
            cls = torch.triu(cls, 1).view(-1)
            sim = torch.from_numpy(sim).float().cuda(non_blocking=True)
            #pdb.set_trace()
            sim = torch.triu(sim, 1).view(-1)
            #pdb.set_trace()
            kl_mean = F.kl_div(cls.softmax(dim=-1).log(), sim.softmax(dim=-1), reduction='sum')

            train_loss = criterion(train_logits, gt_label) + kl_mean
        
        if loss == 'KL2_LOSS':
            sim = np.load('src/sim.npy')
            cls_weight = l2_norm(train_logits, 0)
            cls_weight_t = torch.transpose(cls_weight, 1, 0)
           
            cls = torch.mm(cls_weight_t, cls_weight)
            cls = torch.triu(cls, 1).view(-1)
            sim = torch.from_numpy(sim).float().cuda(non_blocking=True)
            sim = torch.triu(sim, 1).view(-1)
            #pdb.set_trace()
            kl_mean = F.kl_div(cls.softmax(dim=-1).log(), sim.softmax(dim=-1), reduction='sum')

            train_loss = criterion(train_logits, gt_label) + kl_mean
        if loss == 'MAP_LOSS':
            #drivation
            #index = torch.argmax(spa_att)
            #pdb.set_trace()
            #IOU 
            c_2 = torch.sum(mask.float().cuda() * spa_att) 
            
            #l1 norm
            c_3 =   torch.sum(spa_att).float().cuda()
           
            train_loss = criterion(train_logits, gt_label) - torch.log(c_2) + 1.2*torch.log(c_3)
            
            
            
        if loss == 'BCE_LOSS':
            '''
            for name, m in model.named_modules():
                #print(name)
                #'name' can be used to exclude some specified layers
                if name=='module.classifier.fc_1':
                    #pdb.set_trace()
                    #for i in [1,6,12,19,22,23,28,29,31,35,40,42,45,50,52,58,60]:
                    #   y_cov = y_cov + output_0[:,i,:,:]                    
                    mma = get_mma_loss(m.weight)
            '''
            #pdb.set_trace()
            #mm = torch.stack( [torch.sum(cha_att, dim=0), torch.sum(spa_att, dim=0)], dim=0)
            #mma = get_mma_loss(mm)            
            train_loss = criterion(train_logits , gt_label ) #+0.5*mma#+ 0.5*criterion(cha_att, gt_label) #+ 0.02*torch.abs((32-torch.sum(torch.abs(cha_att))))         
            #train_loss = criterion(train_logits * (mask.expand_as(train_logits)), gt_label * (mask.expand_as(gt_label))) #+0.5*mma#+ 0.5*criterion(cha_att, gt_label) #+ 0.02*torch.abs((32-torch.sum(torch.abs(cha_att))))
        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))
 
        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        #gt_list.append(gt_label[:, 6:].cpu().numpy())
        #train_probs = torch.sigmoid(train_logits[:, 6:])
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
    grad_cam = GradCam(model=model, target_layer_names=["layer4"], use_cuda=True)

    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    if True:
    #with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            #pdb.set_trace()
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            #gt_list.append(gt_label[:, 6:].cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits = model(imgs)
            
            #mask_cam = grad_cam(imgs, 22)#mask: bs, 256, 192
            valid_loss = criterion(valid_logits , gt_label)#+ 0.2*(torch.sum(torch.abs(cha_att))+torch.sum(torch.abs(spa_att)))           
            #valid_loss = criterion(valid_logits * (mask.expand_as(valid_logits)), gt_label * (mask.expand_as(gt_label)))#+ 0.2*(torch.sum(torch.abs(cha_att))+torch.sum(torch.abs(spa_att)))
            valid_probs = torch.sigmoid(valid_logits)           
            #valid_probs = torch.sigmoid(valid_logits[:, 6:])
            preds_probs.append(valid_probs.detach().cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))
            #show_att(imgname, spa_att,spa_att )
            #affine(imgname, theta)
            #vif(imgname, spa_att, spa_att)
            #show_on_image(imgname, mask_cam, 22, gt_label)
            #return 0
    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
