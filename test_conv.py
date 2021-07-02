import os
import pprint
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from batch_engine_conv import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block_conv import FeatClassifier, BaseClassifier
from models.resnet18_conv import resnet18_conv
from tools.function import  get_model_log_path, get_pedestrian_metrics
from tools.utils import load_ckpt, time_str, save_ckpt, ReDirectSTD, set_seed

set_seed(605)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('load the model from:   ' + args.save_path )
    exp_dir = os.path.join(args.save_path, args.dataset, args.dataset, 'img_model/ckpt_max.pth')
    train_tsfm, valid_tsfm = get_transform(args)
   
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm, target_transform=None, Type='val')
    
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    print('have generated dataset')


    if args.model_name == 'resnet18_conv':
        backbone = resnet18_conv()   
  
    classifier = BaseClassifier(nattr=valid_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(exp_dir)['state_dicts'])
    
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
