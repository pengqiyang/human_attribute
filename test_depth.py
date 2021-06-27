import os
import pprint
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import pdb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import argument_parser

from loss.CE_loss import CEL_Sigmoid

from models.resnet import resnet50, resnet18, resnet34
from models.resnet18_depth import resnet18_depth
from tools.function import  get_model_log_path, get_pedestrian_metrics
from tools.utils import load_ckpt, time_str, save_ckpt, ReDirectSTD, set_seed
from models.ACNet_models_V1 import resnet18_acnet
from models.resnet18_attention_depth import resnet18_attention_depth
from models.resnet18_no_attention_depth import resnet18_no_attention_depth
from models.resnet_attention_depth_spatial import resnet_attention_depth_spatial
from models.resnet_depth_selective_fusion import resnet_depth_selective_fusion
from models.resnet_attention_depth_cbam_spatial import resnet_attention_depth_cbam_spatial
from dataset.AttrDataset_depth_split import AttrDataset, get_transform
from batch_engine_depth import valid_trainer, batch_trainer
from models.base_block_depth import FeatClassifier, BaseClassifier
from models.resnet18_inception_depth_4 import resnet18_inception_depth_4
from models.resnet18_self_attention_depth_34 import resnet18_self_attention_depth_34
from models.resnet18_self_attention_depth_34_version2 import resnet18_self_attention_depth_34_version2
from models.resnet18_inception_depth_4_wrap import resnet18_inception_depth_4_wrap
from models.ours import ours
from models.resnet_depth import resnet_depth
from models.resnet_attention import resnet_attention 
from models.resnet18_self_mutual_attention import resnet18_self_mutual_attention
'''
from batch_engine import valid_trainer, batch_trainer
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet18_depth import resnet18_depth
from dataset.AttrDataset_depth import AttrDataset, get_transform
'''
set_seed(605)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('load the model from:   ' + args.save_path )
    exp_dir = os.path.join(args.save_path, args.dataset, args.dataset, 'img_model/ckpt_max.pth')
    train_tsfm, valid_tsfm = get_transform(args)
    #pdb.set_trace()
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
    if args.model_name == 'resnet18_depth':
        backbone = resnet18_depth()
    if args.model_name == 'acnet':
        backbone = resnet18_acnet( num_classes = valid_set.attr_num)    
    if args.model_name == 'resnet18_attention_depth':
        backbone = resnet18_attention_depth( num_classes = valid_set.attr_num)    
    if args.model_name == 'resnet18_no_attention_depth':
        backbone = resnet18_no_attention_depth( num_classes = valid_set.attr_num)
    if args.model_name == 'resnet_depth_selective_fusion':
        backbone = resnet_depth_selective_fusion( num_classes = valid_set.attr_num)          
    if args.model_name == 'resnet_attention_depth_spatial':
        backbone = resnet_attention_depth_spatial( num_classes = valid_set.attr_num)  
    if args.model_name == 'resnet_attention_depth_cbam_spatial':
        backbone = resnet_attention_depth_cbam_spatial( num_classes = valid_set.attr_num) 
    if args.model_name == 'resnet18_inception_depth_4':
        backbone = resnet18_inception_depth_4( num_classes = valid_set.attr_num)
    if args.model_name == 'resnet18_self_attention_depth_34':    
        backbone = resnet18_self_attention_depth_34( num_classes = valid_set.attr_num)
    if args.model_name == 'resnet_attention':    
        backbone = resnet_attention(num_classes = valid_set.attr_num)       
    if args.model_name == 'ours':
        backbone = ours( num_classes = valid_set.attr_num)    
    if args.model_name == 'resnet18_self_attention_depth_34_version2':
        backbone = resnet18_self_attention_depth_34_version2( num_classes = valid_set.attr_num)         
    if args.model_name == 'resnet18_inception_depth_4_wrap':
        backbone = resnet18_inception_depth_4_wrap( num_classes = valid_set.attr_num)
    if args.model_name == 'resnet18_self_mutual_attention':
        backbone = resnet18_self_mutual_attention( num_classes = valid_set.attr_num)        
    if args.model_name == 'resnet_depth':
        backbone = resnet_depth( num_classes = valid_set.attr_num)        
    classifier = BaseClassifier(nattr=valid_set.attr_num)
    classifier_depth = BaseClassifier(nattr=valid_set.attr_num)
    model = FeatClassifier(backbone, classifier)
    #model = FeatClassifier(backbone, classifier, classifier_depth)
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
