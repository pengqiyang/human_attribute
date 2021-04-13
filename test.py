import os
import pprint
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet18
<<<<<<< HEAD
from models.resnet_se import resnet50_dynamic_se
from tools.function import  get_model_log_path, get_pedestrian_metrics
from tools.utils import load_ckpt, time_str, save_ckpt, ReDirectSTD, set_seed

from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
>>>>>>> 7c1b54e1c9904b204dd8732a426f14daa05b7f7b

set_seed(605)


def main(args):
<<<<<<< HEAD
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

 
>>>>>>> 7c1b54e1c9904b204dd8732a426f14daa05b7f7b
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
<<<<<<< HEAD
    print('have generated dataset')

>>>>>>> 7c1b54e1c9904b204dd8732a426f14daa05b7f7b
    if args.model_name == 'resnet50':
        backbone = resnet50()
    if args.model_name == 'resnet18':
        backbone = resnet18()
    
<<<<<<< HEAD
    classifier = BaseClassifier(nattr=valid_set.attr_num)

>>>>>>> 7c1b54e1c9904b204dd8732a426f14daa05b7f7b
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
<<<<<<< HEAD
   

    #loading state_dict from the model
    model.load_state_dict(torch.load(exp_dir)['state_dicts'])
    #load_ckpt(model, exp_dir)
    print('have load from the pretrained model')
=======



>>>>>>> 7c1b54e1c9904b204dd8732a426f14daa05b7f7b
    #start eval
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