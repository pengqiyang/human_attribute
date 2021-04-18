#python3 test.py PA100k --device='0,1,2,3' --model_name='resnet50' --save_path='' --train_split='train' --valid_split='test' 
#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='0' --save_path='checkpoints/resnet18_batch64' --model_name='resnet18' --train_split='train' --valid_split='test'


#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4' --save_path='checkpoints/resnet50_dynamic_se_batch16' --model_name='resnet50_dynamic_se' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4' --save_path='resnet18_se' --model_name='resnet18_dynamic_se' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL2_LOSS' --device='4' --save_path='resnet18_kl2' --model_name='resnet18' --train_split='train' --valid_split='test'

python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4' --save_path='resnet18_vit_v5' --model_name='resnet18_vit_v5' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4' --save_path='resnet34' --model_name='resnet34' --train_split='train' --valid_split='test'
