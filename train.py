import os
import pprint
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import pdb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from models.resnet_depth import resnet_depth
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet18, resnet34
from models.resnet50_dynamic_se import resnet50_dynamic_se
from models.resnet18_dynamic_se import resnet18_dynamic_se
from models.resnet18_replace_se import resnet18_replace_se
from models.resnet_se import resnet18_se
from models.resnet18_energy_vit import resnet18_energy_vit
from models.resnet18_vit import resnet18_vit
from models.resnet18_vit_v2 import resnet18_vit_v2
from models.resnet18_vit_v3 import resnet18_vit_v3
from models.resnet18_vit_v4 import resnet18_vit_v4
from models.resnet18_vit_v5 import resnet18_vit_v5
from models.resnet18_group_se import resnet18_group_se
from models.resnet18_vit_split import resnet18_vit_split
from models.resnet18_energy_vit import resnet18_energy_vit
from models.inception_self import inception_self
from models.resnet_stn import resnet18_stn
from models.resnet18_transformer import resnet18_transformer
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
from models.spatial_modulator import spatial_modulator
from models.fusion_concat import fusion_concat
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
    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    #train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)
    train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm, target_transform=None, Type='train' )
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    #valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm, target_transform=None, Type='val')
    
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    #backbone = resnet50()
    if args.model_name == 'resnet50':
        backbone = resnet50()
    if args.model_name == 'resnet18':
        backbone = resnet18()
    if args.model_name == 'resnet18_stn':
        backbone = resnet18_stn()       
    if args.model_name == 'resnet_depth':
        backbone = resnet_depth()
    if args.model_name == 'resnet18_transformer':
        backbone = resnet18_transformer()        
    if args.model_name == 'resnet50_dynamic_se':
        backbone = resnet50_dynamic_se()
    if args.model_name == 'resnet18_dynamic_se':
        backbone = resnet18_dynamic_se()
    if args.model_name == 'resnet18_replace_se':
        backbone = resnet18_replace_se()
    if args.model_name == 'resnet18_se':
        backbone = resnet18_se()
    if args.model_name == 'resnet34':
        backbone = resnet34()
    if args.model_name == 'resnet18_group_se':
        backbone = resnet18_group_se()
    if args.model_name == 'resnet18_vit':
        backbone = resnet18_vit()
    if args.model_name == 'resnet18_vit_v2':
        backbone = resnet18_vit_v2()
    if args.model_name == 'resnet18_vit_v3':
        backbone = resnet18_vit_v3()
    if args.model_name == 'resnet18_vit_v5':
        backbone = resnet18_vit_v5()
    if args.model_name == 'resnet18_energy_vit':
        backbone = resnet18_energy_vit()
    if args.model_name == 'resnet18_vit_split':
        backbone = resnet18_vit_split(num_classes = train_set.attr_num)
    if args.model_name == 'inception_self':
        backbone = inception_self()  
    if args.model_name == 'spatial_modulator':
        backbone = spatial_modulator()
    if args.model_name == 'fusion_concat':
        backbone = fusion_concat()         
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
    '''
    model_dict = {}
    state_dict = model.state_dict()
    pretrain_dict = torch.load('/home/pengqy/paper/resnet18_2/PETA/PETA/img_model/ckpt_max.pth')['state_dicts']
    for k, v in pretrain_dict.items():
        # print('%%%%% ', k)
        if k in state_dict:
            if k.startswith('module.backbone.conv1'):
                #pdb.set_trace()
                model_dict[k] = v       
            elif k.startswith('module.backbone.bn1'):
                model_dict[k] = v          
            elif k.startswith('module.backbone.layer'):
                model_dict[k] = v
            elif k.startswith('module.classifier'):
                model_dict[k] = v
            
            #elif k.startswith('module.backbone.spa_conv_0'):
            #    model_dict[k] = v
            #elif k.startswith('module.backbone.spa_bn_0'):
            #    model_dict[k] = v 
            #elif k.startswith('module.classifier'):
            #    model_dict[k] = v
            #elif k.startswith('module.classifier'):
            #    model_dict[k] = v   
              
    #pdb.set_trace()       
         
    state_dict.update(model_dict) 
    model.load_state_dict(state_dict)
    '''
    criterion = CEL_Sigmoid(sample_weight)
   
  
    
    param_groups = [{'params': model.module.finetune_params(), 'lr':0.01},
                   {'params': model.module.fresh_params(), 'lr':0.1}]
    
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
        save_ckpt(model, path, i, maximum)    

        result_list[i] = [train_result, valid_result]

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
