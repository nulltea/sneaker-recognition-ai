import os;
import cv2;
import random;
import torch
import data_transform;
import numpy as np;
import matplotlib.pyplot as plt;
from PIL import Image;
from torch.utils.data import Dataset;
from torchvision import transforms, utils;
from config import *;

class SneakersDataset(Dataset):
	"""Sneakers dataset."""
	def __init__(self, img_dir, labels, transform_functions=None):
		"""
		Args:
			img_dir (string): Directory with all the images.
			transforms ([callable], optional): Optional transform to be applied
				on a sample.
		"""
		self.img_dir = img_dir
		self.labels = labels;
		self.image_arr = data_transform.images_to_array(img_dir, labels)
		self.transform_functions = transform_functions;
		self.to_tensor = transforms.ToTensor()

	def __len__(self):
		return len(self.image_arr)

	def __getitem__(self, index):
		image, label = self.image_arr[index];
		img_as_tensor = self.to_tensor(image);
		if self.transform_functions and any(self.transform_functions):
			for transform in self.transform:
				transform(img_as_tensor);
		return (img_as_tensor, label);

def select_n_random(dataset, n=100):
	perm = torch.randperm(len(dataset))
	return [img for img,_ in dataset[perm][:n]], [label for _,label in dataset[perm][:n]];


if __name__ == "__main__":
	from config import *;
	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter('runs/sneaker_net_test')
	dataset = SneakersDataset(IMG_DIR, MODELS);

	# select random images and their target indices
	images, labels = select_n_random(dataset)

	# get the class labels for each image
	class_labels = [MODELS[lab] for lab in labels]

	# log embeddings
	features = images.view(-1, 28 * 28)
	writer.add_embedding(features,
						metadata=class_labels,
						label_img=images.unsqueeze(1))
	writer.close()

	for i in range(20):
		arr, label = dataset[0]
		img_grid = utils.make_grid(arr)
		# write to tensorboard
		writer.add_image('sneaker_net_img', img_grid);
	input();
	writer.close();