#!/usr/bin/env bash
#python3 train.py PA100k --batchsize=64 --loss='BCE_LOSS' --device='1' --save_path='resnet18_dynamic_se' --model_name='resnet18_dynamic_se' --train_split='train' --valid_split='val'

#python3 train.py PA100k --loss='KL_LOSS' --device='0' --save_path='resnet50_kl' --model_name='resnet50' --train_split='train' --valid_split='val'

python3 train.py PA100k --batchsize=16 --loss='BCE_LOSS' --device='0' --save_path='resnet50' --model_name='resnet50' --train_split='train' --valid_split='val'

#python3 train.py PA100k --loss='BCE_LOSS' --device='0' --save_path='resnet50_dynamic_se' --model_name='resnet50_dynamic_se' --train_split='train' --valid_split='val'
