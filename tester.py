import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

import service
from config import *
from data_transform import *
from model import *
from resnet import *


def img_show(img, text):
	# img = img / 2 + 0.5		# unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


activation = {}


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook


tensorboard = SummaryWriter('runs/sneaker_net');
net = Net();  # resnet101(3, len(MODELS));
net.load_state_dict(torch.load(MODEL_SAVE_PATH));
net.conv_layers[0].register_forward_hook(get_activation('conv'))

batch_size = 4;
conv_count = 4;
trainset = [];

for image in os.listdir(TEST_DIR):
	img_array = image_tranform_to_tensor(os.path.join(TEST_DIR, image));
	if img_array is None:
		continue;
	trainset.append(img_array);
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True)
dataiter = iter(dataloader)
data_batch = dataiter.next();
outputs = net(data_batch);
values, labels = torch.max(outputs, 1);
predicted = [MODELS[labels[j]] for j in range(batch_size)]

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, ax in enumerate(axs.flatten()):
	plt.sca(ax)
	npimg = data_batch[i].numpy();
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.title(predicted[i]);
plt.suptitle('Predictions:')
plt.show()

j = 0
f, axarr = plt.subplots(batch_size, conv_count, figsize=(8, 6))
for layer in range(conv_count):
	convx = "conv"
	act = activation[convx].squeeze();
	for img in range(batch_size):
		axarr[img, layer].imshow(act[img][layer]);
plt.show()

# TODO make gui or io.web app
# use image transform visualization as loading...
#
