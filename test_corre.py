import os
import pprint
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from batch_engine_corre import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block_corre import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet18, resnet34
from models.resnet50_dynamic_se import resnet50_dynamic_se
from models.resnet18_dynamic_se import resnet18_dynamic_se
from models.resnet18_group_se import resnet18_group_se
from models.resnet18_vit import resnet18_vit
from models.resnet18_vit_v2 import resnet18_vit_v2
from models.resnet18_vit_v3 import resnet18_vit_v3
from models.resnet18_vit_v5 import resnet18_vit_v5
from models.resnet18_vit_split import resnet18_vit_split
from models.resnet18_energy_vit import resnet18_energy_vit
from models.resnet18_autoencoder import resnet18_autoencoder
from tools.function import  get_model_log_path, get_pedestrian_metrics
from tools.utils import load_ckpt, time_str, save_ckpt, ReDirectSTD, set_seed
from models.resnet_depth import resnet_depth
set_seed(605)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('load the model from:   ' + args.save_path )
    exp_dir = os.path.join(args.save_path, args.dataset, args.dataset, 'img_model/ckpt_max.pth')
    train_tsfm, valid_tsfm = get_transform(args)
   
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print('have generated dataset')

    if args.model_name == 'resnet50':
        backbone = resnet50()
    if args.model_name == 'resnet18':
        backbone = resnet18()
    if args.model_name == 'resnet18_autoencoder':
        backbone = resnet18_autoencoder()
    if args.model_name == 'resnet50_dynamic_se':
        backbone = resnet50_dynamic_se()
    if args.model_name == 'resnet18_dynamic_se':
        backbone = resnet18_dynamic_se()
    if args.model_name == 'resnet18_group_se':
        backbone = resnet18_group_se()
    if args.model_name == 'resnet18_vit':
        backbone = resnet18_vit()
    if args.model_name == 'resnet18_vit_v2':
        backbone = resnet18_vit_v2()
    if args.model_name == 'resnet18_vit_v3':
        backbone = resnet18_vit_v3()
    if args.model_name == 'resnet18_vit_v4':
        backbone = resnet18_vit_v4()
    if args.model_name == 'resnet34':
        backbone = resnet34()        
    if args.model_name == 'resnet18_vit_split':
        backbone = resnet18_vit_split(num_classes = valid_set.attr_num)
    if args.model_name == 'resnet18_energy_vit':
        backbone = resnet18_energy_vit(num_classes = valid_set.attr_num)
    if args.model_name == 'resnet_depth':
        backbone = resnet_depth( num_classes = valid_set.attr_num)        
    classifier = BaseClassifier(nattr=valid_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        #model = model.cuda()

    #loading state_dict from the model
    model.load_state_dict(torch.load(exp_dir)['state_dicts'])
    
    #load_ckpt(model, exp_dir)
    print('have load from the pretrained model')
    

    #start eval
    labels = valid_set.label
    sample_weight = labels.mean(0)
    criterion = CEL_Sigmoid(sample_weight)
    valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )
    valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

    #print result
    print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

    for index in range(len(valid_set.attr_name)):
        print(f'{valid_set.attr_name[index]}')
        print(f'pos recall: {valid_result.label_pos_recall[index]}  neg_recall: {valid_result.label_neg_recall[index]}  ma: {valid_result.label_ma[index]}')
        

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
