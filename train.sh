i#!/usr/bin/env bash
#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='1' --save_path='resnet18_dynamic_se' --model_name='resnet18_dynamic_se' --train_split='train' --valid_split='val'

#python3 train.py PA100k --loss='KL_LOSS' --device='0' --save_path='resnet50_kl' --model_name='resnet50' --train_split='train' --valid_split='val'

#python3 train.py PA100k --batchsize=16 --loss='BCE_LOSS' --device='0' --save_path='resnet50' --model_name='resnet50' --train_split='train' --valid_split='val'

#python3 train.py PA100k --loss='BCE_LOSS' --device='0' --save_path='resnet50_dynamic_se' --model_name='resnet50_dynamic_se' --train_split='train' --valid_split='val'

#python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='4' --save_path='resnet50' --model_name='resnet50' --train_split='train' --valid_split='val'



#2021.4.14

#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='4' --save_path='resnet18' --model_name='resnet18' --train_split='train' --valid_split='val'

#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='5' --save_path='resnet18_se' --model_name='resnet18_dynamic_se' --train_split='train' --valid_split='val'

#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='4' --save_path='resnet18' --model_name='resnet34' --train_split='train' --valid_split='val'

#2021.4.15
#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='4' --save_path='resnet18_vit_v2' --model_name='resnet18_vit_v2' --train_split='train' --valid_split='val'

#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='5' --save_path='resnet18_vit_v4' --model_name='resnet18_vit_v4' --train_split='train' --valid_split='val'


#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='resnet18_vit_split' --model_name='resnet18_vit_split' --train_split='train' --valid_split='val'

#python3 train.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4,5' --save_path='resnet18_vit_v5' --model_name='resnet18_vit_v5' --train_split='train' --valid_split='val'

#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='4' --save_path='resnet34' --model_name='resnet34' --train_split='train' --valid_split='val'
#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='4' --save_path='resnet34' --model_name='resnet34' --train_split='train' --valid_split='val'


python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='6' --save_path='/home/pengqy/paper/resnet18_0' --model_name='resnet18' --train_split='train' --valid_split='val'
python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='6' --save_path='/home/pengqy/paper/resnet18_1' --model_name='resnet18' --train_split='train' --valid_split='val'
python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='6' --save_path='/home/pengqy/paper/resnet18_2' --model_name='resnet18' --train_split='train' --valid_split='val'
python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='6' --save_path='/home/pengqy/paper/resnet18_3' --model_name='resnet18' --train_split='train' --valid_split='val'
#python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='4,5,6,7' --save_path='/home/pengqy/paper/resnet18_concat_4' --model_name='fusion_concat' --train_split='train' --valid_split='val'

#python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_spatial_0' --model_name='resnet18' --train_split='train' --valid_split='val'
#python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_spatial_1' --model_name='resnet18' --train_split='train' --valid_split='val'
#python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_spatial_2' --model_name='resnet18' --train_split='train' --valid_split='val'
#python3 train.py PETA --batchsize=64 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_spatial_3' --model_name='resnet18' --train_split='train' --valid_split='val'
