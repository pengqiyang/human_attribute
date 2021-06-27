import time
import torch.nn.functional as F 
import pdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time
import torch.nn.functional as F 
import pdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from visualization.vis_feature_map import vif
from tools.utils import AverageMeter, to_scalar, time_str
import torch.nn as nn
import torch.nn.functional as F


loss_autoencoder = nn.SmoothL1Loss()
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
    for step, (imgs, gt_label, gt_depth, imgname) in enumerate(train_loader):
        #print(step)
        batch_time = time.time()
        imgs, gt_label, gt_depth = imgs.cuda(), gt_label.cuda(), gt_depth.cuda()
        train_logits, _ = model(imgs)
        #print("sssssssssssssssssssssssssssss")
        #pdb.set_trace()
        if loss == 'BCE_LOSS':
            train_loss = criterion(train_logits, gt_label) + loss_autoencoder(_, gt_depth)
        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
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
        for step, (imgs, gt_label, gt_depth, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_depth =  gt_depth.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits,_ = model(imgs)
            valid_loss = criterion(valid_logits, gt_label) + loss_autoencoder(_, gt_depth)
            valid_probs = torch.sigmoid(valid_logits)
           
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
