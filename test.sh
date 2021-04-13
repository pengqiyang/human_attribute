#python3 test.py PA100k --device='0,1,2,3' --model_name='resnet50' --save_path='' --train_split='train' --valid_split='test' 
python3 test.py PA100k --loss='KL_LOSS' --device='0,1' --save_path='resnet50_kl' --model_name='resnet50' --train_split='train' --valid_split='test'
