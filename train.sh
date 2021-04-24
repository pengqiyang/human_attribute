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

python3 train.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4,5' --save_path='resnet18_vit_v5' --model_name='resnet18_vit_v5' --train_split='train' --valid_split='val'

#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='4' --save_path='resnet34' --model_name='resnet34' --train_split='train' --valid_split='val'
python3 train_depth.py PA100k --batchsize=64 --loss='KL_LOSS' --device='0' --save_path='resnet18_depth_KL' --model_name='resnet18' --train_split='train' --valid_split='val'
