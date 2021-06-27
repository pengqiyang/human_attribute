#python3 test.py PA100k --device='0,1,2,3' --model_name='resnet50' --save_path='' --train_split='train' --valid_split='test' 
#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='0' --save_path='checkpoints/resnet18_batch64' --model_name='resnet18' --train_split='train' --valid_split='test'


#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4' --save_path='checkpoints/resnet50_dynamic_se_batch16' --model_name='resnet50_dynamic_se' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='4' --save_path='resnet18_se' --model_name='resnet18_dynamic_se' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL2_LOSS' --device='4' --save_path='resnet18_kl2' --model_name='resnet18' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='3' --save_path='resnet18_vit_split' --model_name='resnet18_vit_split' --train_split='train' --valid_split='test'

#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='1,3' --save_path='resnet34' --model_name='resnet34' --train_split='train' --valid_split='test'
#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='1,3' --save_path='resnet18' --model_name='resnet18' --train_split='train' --valid_split='test'
#python3 test.py PA100k --batchsize=64 --loss='KL_LOSS' --device='0,1,2,3' --save_path='resnet18_vit_split' --model_name='resnet18_vit_split' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=64 --loss='KL_LOSS' --device='3' --save_path='resnet_depth' --model_name='resnet18_depth' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='3' --save_path='acnet' --model_name='acnet' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='4' --save_path='resnet_attention_depth_layer2' --model_name='resnet18_attention_depth' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='3' --save_path='resnet18_no_attention_depth' --model_name='resnet18_no_attention_depth' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='4,5,6,7' --save_path='resnet_attention_depth_cbam_spatial' --model_name='resnet_attention_depth_cbam_spatial' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='4,5,6,7' --save_path='resnet_depth_selective_fusion_4567' --model_name='resnet_depth_selective_fusion' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='4,5,6,7' --save_path='resnet18_attention_depth_34' --model_name='resnet18_attention_depth' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='4,5,6,7' --save_path='resnet18_inception_34_seblock' --model_name='resnet18_inception_depth_4' --train_split='train' --valid_split='test'
#python3 test_depth.py PA100k --batchsize=16 --loss='KL_LOSS' --device='4,5,6,7' --save_path='resnet18_inception_34_seblock' --model_name='resnet18_self_attention_depth_34_version2' --train_split='train' --valid_split='test'

python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_concat_0' --model_name='fusion_concat' --train_split='train' --valid_split='test'
python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_concat_1' --model_name='fusion_concat' --train_split='train' --valid_split='test'
python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_concat_2' --model_name='fusion_concat' --train_split='train' --valid_split='test'
python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_concat_3' --model_name='fusion_concat' --train_split='train' --valid_split='test'
python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='/home/pengqy/paper/resnet18_concat_4' --model_name='fusion_concat' --train_split='train' --valid_split='test'

#python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='paper/resnet18_0' --model_name='resnet18' --train_split='train' --valid_split='test'
#python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='paper/resnet18_1' --model_name='resnet18' --train_split='train' --valid_split='test'
#python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='paper/resnet18_2' --model_name='resnet18' --train_split='train' --valid_split='test'
#python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='paper/resnet18_3' --model_name='resnet18' --train_split='train' --valid_split='test'

#python3 test.py  PETA  --batchsize=16 --loss='BCE_LOSS' --device='0,1,2,3' --save_path='paper/resnet18_4' --model_name='resnet18' --train_split='train' --valid_split='test'
