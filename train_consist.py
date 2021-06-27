import os
import pprint
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import pdb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from bacth_engine_consist import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDatasetConsist import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block_consist import FeatClassifier, BaseClassifier
from models.resnet18_consistent import resnet18_consistent
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
'''
part_init_0 =[]
part_init_1 =[]
part_init_2 =[]

for i in range(35):
     part_init_0.append(torch.from_numpy(np.load('tools/layer3/'+str(i)+'.npy', allow_pickle=True))[0].unsqueeze(0))
     part_init_1.append(torch.from_numpy(np.load('tools/layer3/'+str(i)+'.npy', allow_pickle=True))[1].unsqueeze(0))
     part_init_2.append(torch.from_numpy(np.load('tools/layer3/'+str(i)+'.npy', allow_pickle=True))[2].unsqueeze(0))
#pdb.set_trace()
#part_init = [i.numpy() for i in (list(part_init))]
part_init_0 = torch.cat(part_init_0).cuda().float()
part_init_1 = torch.cat(part_init_1).cuda().float()
part_init_2 = torch.cat(part_init_2).cuda().float()
part_init_0 = torch.nn.functional.normalize(part_init_0, p=2, dim=1, eps=1e-12, out=None)#35, 256,1 
part_init_1 = torch.nn.functional.normalize(part_init_1, p=2, dim=1, eps=1e-12, out=None)#35, 256,1  
part_init_2 = torch.nn.functional.normalize(part_init_2, p=2, dim=1, eps=1e-12, out=None)#35, 256,1        
#part_init = torch.from_numpy(np.array(part_init)).float()
'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

set_seed(605)

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join(args.save_path, args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')
   
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    #train_tsfm, valid_tsfm = get_transform(args)
    train_tsfm, train_tsfm_resize, valid_tsfm, valid_tsfm_resize = get_transform(args)
    print(train_tsfm)

    train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm, transform_resize = train_tsfm_resize )
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    #valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm, transform_resize = valid_tsfm_resize)
    
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    if args.model_name == 'resnet18_consistent':
        backbone = resnet18_consistent()
       
    print('have generated the model')    
    classifier = BaseClassifier(nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)
    
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('')
    
    #if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    #for k, v in model.state_dict().items():
    #    print(k)
    
    
    model_dict = {}
    state_dict = model.state_dict()
    #pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
    pretrain_dict = torch.load('ckpt_max_23.pth')['state_dicts']
    for k, v in pretrain_dict.items():
        #print(k)
        #if k in state_dict:
                          
        #if k.startswith('module.backbone.conv1'):
        #if k.startswith('module.backbone.fix'):
            #pdb.set_trace()
        #    model_dict['module.backbone'+k[19:]] = v 
        #    #model_dict['module.backbone.fix'+k[15:]] = v       
        if k.startswith('module.backbone.'):
        #    #pdb.set_trace()
            model_dict[k] = v
        
        if k.startswith('module.classifier.conv'):
            #pdb.set_trace()
            model_dict['module.backbone.conv'+k[22:]] = v
          
        elif k.startswith('module.classifier.bn'):
            #pdb.set_trace()
            model_dict['module.backbone.bn'+k[20:]] = v     
        
       
            
        #elif k.startswith('layer3'):
        #    model_dict['module.backbone.fix.'+k] = v
        #elif k.startswith('layer4'):
        #    model_dict['module.backbone.fix.'+k] = v               
        #elif k.startswith('module.classifier.conv_3'):
        #    model_dict['module.backbone.fix'+k[17:]] = v  
        #elif k.startswith('module.classifier'):
        #    model_dict[k] = v
        #elif k.startswith('module.classifier'):
        #    model_dict[k] = v   
       
    #pdb.set_trace()       
    #model_dict['module.backbone.fix.keypoints_0'] = part_init_0      
    #model_dict['module.backbone.fix.keypoints_1'] = part_init_1      
    #model_dict['module.backbone.fix.keypoints_2'] = part_init_2      
    
    #for k , v in state_dict.items():
    #    print(k)
        
        
    #print('sss')
    #for k, v in state_dict.items():
    #    print(k)
    #pdb.set_trace()
    
    state_dict.update(model_dict) 
    model.load_state_dict(state_dict)
   
    for name, child in model.module.backbone.named_children():
        #print(name)
        if name == 'fix':
            #pdb.set_trace() 
            for param in child.parameters():
                #pdb.set_trace()
                #print('sss')
                param.requires_grad = True    
    
    #pdb.set_trace()
    
    criterion = CEL_Sigmoid(sample_weight)
    #model.load_state_dict(torch.load('/home/pengqy/paper/resnet18_consist/PETA/PETA/img_model/ckpt_max.pth')['state_dicts'])

     
    param_groups = [{'params': model.module.finetune_params(), 'lr':0.0001},
                   #{'params': model.module.stn_params(), 'lr': 0.0001},
                   {'params': model.module.fresh_params(), 'lr': 0.1}]
     
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
    loss = args.loss
    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path,
                                 loss =loss)

    print(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')


def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path, loss):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for i in range(epoch):
        
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            loss = loss,
        )
        

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        lr_scheduler.step(metrics=valid_loss, epoch=i)

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric = valid_result.ma

        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)
        else:
            save_ckpt(model,  os.path.join(path.split('.')[0]+"_"+str(i)+'.pth'), i, maximum)  

        result_list[i] = [train_result, valid_result]

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()
