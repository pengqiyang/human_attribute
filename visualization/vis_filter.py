# -*- coding: utf-8 -*-
import numpy as np
import pdb
from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms


if __name__ == '__main__':

	#model = get_model('densenet121')  # change model what you want
	model = torch.load('/home/pengqy/paper/resnet18_2/PETA/PETA/img_model/ckpt_max.pth')['state_dicts']
	
	weight = model['module.backbone.conv1.weight'][:,1,:,:].unsqueeze(1)

	#pdb.set_trace()
	utils.vis_conv(weight.cpu().detach().numpy(), 8, 8, "filter", "%s_filter_G"%("densenet121"))	

	