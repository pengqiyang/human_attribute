import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import matplotlib.pyplot as plt
from torchsummary import summary
import visualization.utils
import pdb

class FeatureExtractor():
	""" Class for extracting activations and 
	registering gradients from targetted intermediate layers """

	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
	
		def hook_fn(module, grad_in, grad_out):
			self.gradients.append(grad_out[0])
			
		outputs = []
		self.gradients = []
		#for name, child in self.model.named_children():
		for name, module in self.model._modules.items():
			#pdb.set_trace()
			x = module(x)
			#pdb.set_trace()
			print(name)
			if name in self.target_layers:
				#pdb.set_trace()
				module.register_backward_hook(hook_fn)
				x.register_hook(self.save_gradient)
				outputs += [x]
		return outputs, x


class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """

	def __init__(self, model, target_layers):
		self.model = model
		#pdb.set_trace()
		self.feature_extractor = FeatureExtractor(self.model.module.backbone, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output = self.feature_extractor(x)
		#pdb.set_trace()
		#output = output.view(output.size(0), -1)
		output = self.model(x)
		#output = self.model.classifier(output)
		#print(output.size())
		return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	print(img.shape, heatmap.shape)
	heatmap = np.float32(heatmap) / 255

	cam = heatmap + np.float32(img)

	cam = cam / np.max(cam)
	
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.att_num = 51
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index=None):
		#features:????????????CAM??????????????? output????????????????????????
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argsort(output.cpu().data.numpy())[-1][-1]
		
		cam_return = np.zeros((51, input.size()[0], input.size()[2], input.size()[3]), dtype=np.float32)#34, bs, 8, 6		
		for index_attr  in [2,3,6,9,10,12,14,15,16,19,21,22,23,26,27,28]:
			print("index = ", index_attr)
			one_hot = np.zeros((1, self.att_num), dtype=np.float32)
			one_hot[0][index_attr] = 1
			one_hot = torch.from_numpy(one_hot).requires_grad_(True)
			if self.cuda:
				one_hot = torch.sum(one_hot.cuda() * output)
			else:
				one_hot = torch.sum(one_hot * output)
		
			self.model.module.backbone.zero_grad()
			self.model.module.classifier.zero_grad()
			one_hot.backward(retain_graph=True)
			#pdb.set_trace()
			print(len(self.extractor.get_gradients()))
			grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
			print(grads_val.shape)
			print(features[-1].shape)
			target = features[-1]
			target = target.cpu().data.numpy()[:, :]# target:bs, dim ,8, 6
			print(np.mean(grads_val, axis=(2, 3)).shape)
			weights = np.mean(grads_val, axis=(2, 3))[:, :]# weights:bs, dim

			for img_ in range(target.shape[0]):		
				cam = np.zeros(target.shape[2:], dtype=np.float32)
				for i, w in enumerate(weights[img_]):
					cam += w * target[img_, i, :, :]
					#pdb.set_trace()
					cam = np.maximum(cam, 0)

					cam_return[index_attr, img_] = cv2.resize(cam, (input.size()[3], input.size()[2]))
					cam_return[index_attr, img_] = cam_return[index_attr, img_] - np.min(cam_return[index_attr, img_])
					cam_return[index_attr, img_] = cam_return[index_attr, img_] / np.max(cam_return[index_attr, img_])
			#pdb.set_trace()
		return cam_return


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./inputs/dog.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()

	# Can work with any model, but it assumes that the model has a
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	grad_cam = GradCam(model=models.vgg19(pretrained=True), \
					   target_layer_names=["layer4"], use_cuda=args.use_cuda)

	img = cv2.imread(args.image_path, 1)
	#img = np.float32(cv2.resize(img, (224, 224))) / 255
	img = np.float32(img) / 255
	input = preprocess_image(img)

	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
	target_index = None
	mask = grad_cam(input, 1)

	utils.show_image(mask)
	
	show_cam_on_image(img, mask)
