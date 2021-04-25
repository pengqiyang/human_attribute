import time
import torch.nn.functional as F 
import pdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.utils import AverageMeter, to_scalar, time_str

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

    for step, (imgs, depth, gt_label, imgname) in enumerate(train_loader):
        #print(step)
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits = model(imgs, depth, gt_label)
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
        if loss == 'BCE_LOSS':
            train_loss = criterion(train_logits, gt_label) 
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
        for step, (imgs, depth, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits = model(imgs, depth)
            valid_loss = criterion(valid_logits, gt_label)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
