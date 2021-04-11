#!/usr/bin/env bash


python3 train.py PA100k --device=0,1,2,3 --model_name='resnet50' --train_split='train' --valid_split='val'
