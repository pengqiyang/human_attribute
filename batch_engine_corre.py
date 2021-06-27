import time
import torch.nn.functional as F 
import pdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from visualization.vis_feature_map import vif, show_on_image
from tools.utils import AverageMeter, to_scalar, time_str
from models.MMA import *
mse_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
fc_loss = np.load('tools/corre_peta_gongxian.npy')
mask = torch.from_numpy(np.load('tools/mask_peta_gongxian.npy', allow_pickle=True)).float().cuda()
class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D).cuda()).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        #c_diff[~(torch.eye(D).bool())] *= self.lambda_param
        
        c_diff_2 = (c_diff[~(torch.eye(D).bool())]*self.lambda_param).sum()
        loss = c_diff.sum() + c_diff_2

        return loss
twin_loss  = BarlowTwinsLoss()
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
        if step>=319:
            continue
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits, train_logits_2 = model(imgs, gt_label)
        #train_logits = model(imgs, gt_label)
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
            
            train_loss = criterion(train_logits, gt_label) #+ criterion(train_logits_2, gt_label) 
        if loss == 'MMA_LOSS':
            #pdb.set_trace()
            #label_att = np.load('tools/att_label.npy')
            '''
            for name, m in model.named_modules():
                #print(name)
                #'name' can be used to exclude some specified layers
                if name=='module.classifier.fc_1':
                    #pdb.set_trace()
                    #cal_corre  = np.corrcoef(m.weight.cpu().detach().numpy())
                    #cal_corre = torch.from_numpy(cal_corre).float().cuda()
                    #weight_ = F.normalize(m.weight, p=2, dim=1)                 
                    #cosine = torch.matmul(weight_, weight_.t())
                    
                    mma_2 = mse_loss_fn((m.weight).view(-1), ((torch.from_numpy(fc_loss).float().cuda())).view(-1))
                    #mma = twin_loss(m.weight.cuda(), torch.from_numpy(fc_loss).float().cuda())
            #pdb.set_trace()    
            '''
            #train_loss = criterion(train_logits, gt_label) + F.kl_div(torch.mean(train_logits_2.squeeze(), 0)[0], torch.from_numpy(label_att).float().cuda(), reduction='sum')
            #train_loss = criterion(train_logits, gt_label) + 0.3*mse_loss_fn(torch.mean(train_logits_2.squeeze(), 0)[0], torch.from_numpy(label_att).float().cuda())
            train_loss = criterion(train_logits_2, gt_label)#+  mma_2 + criterion(train_logits_2, gt_label) #+ 20*mma_2
            '''
            train_pro =(torch.sigmoid(train_logits)>0.5).float()
            train_loss_2 = torch.matmul((train_loss_2*train_pro).t(), train_loss_2*train_pro)*mask
            train_loss = 0.5*torch.sum(train_loss_2)/train_logits.size()[0] + train_loss_1
            '''
        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        
        #train_probs = torch.sigmoid(train_logits)
        train_probs_2 = torch.sigmoid(train_logits_2)
        
        
        preds_probs.append(train_probs_2.detach().cpu().numpy())

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
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            #valid_logits = model(imgs)
            valid_logits, valid_logits_2 = model(imgs)
            #pdb.set_trace()
            #valid_logits = model(imgs)
            #valid_loss = criterion(valid_logits, gt_label) #+ criterion(valid_logits_2, gt_label) 
            #valid_loss = criterion(valid_logits, gt_label) #+ F.kl_div(torch.mean(valid_logits_2.squeeze(), 0)[0], torch.from_numpy(label_att).float().cuda(), reduction='sum')
            valid_loss = criterion(valid_logits_2, gt_label)#+criterion(valid_logits_2, gt_label) #+ 0.3*mse_loss_fn(torch.mean(valid_logits_2.squeeze()[0], 0), torch.from_numpy(label_att).float().cuda())
            valid_probs = torch.sigmoid(valid_logits_2)
            '''
            valid_probs_2 = torch.sigmoid(valid_logits_2)
            #pdb.set_trace()
            # accessory
            
            valid_prob = valid_probs > 0.5
            for i in range(valid_logits.size()[0]):
                if  (valid_prob[i,15] and valid_prob[i,16]):
                    print(str(valid_probs[i,15]) + ' '+str(valid_probs[i,16]))              
            '''                 
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))
            #show_on_image(imgname, output)      
            #vif(imgname,output_depth_0,output_depth_1,output_depth_2,output_depth_3, output_depth_4, output_depth_5)
            #return 0
            
    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
