#python3 test.py PA100k --device='0,1,2,3' --model_name='resnet50' --save_path='' --train_split='train' --valid_split='test' 
python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='0' --save_path='checkpoints/resnet18_batch64' --model_name='resnet18' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='0,1' --save_path='checkpoints/resnet50_kl_batch16' --model_name='resnet50' --train_split='train' --valid_split='test'
