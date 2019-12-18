import os;
import cv2;
import random;
import torch
import data_transform;
import numpy as np;
import matplotlib.pyplot as plt;
from PIL import Image;
from torch.utils.data import Dataset;
from torchvision import transforms, utils

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

from config import *;
if __name__ == "__main__":
	dataset = SneakersDataset(IMG_DIR, MODELS);
	for i in range(10):
		print(dataset[0]);
	input();