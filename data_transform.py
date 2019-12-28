import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from console_progressbar import ProgressBar
from PIL import Image
from torchvision import transforms, utils

import service

preprocess = transforms.Compose([
	transforms.Resize((128, 128)),
	transforms.CenterCrop((128, 128)),
	transforms.ToTensor(),
	# transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
])


def resize_multiple_images(src_path, dst_path):
	"""
			Args:
					src_path (string): location where images are saved.
					src_path (string): location where images will be saved.
	"""
	for subdir, brands, _ in os.walk(src_path):
		for brand in brands:
			for subdir, models, _ in os.walk(os.path.join(subdir, brand)):
				for model in models:
					model_from_dir = os.path.join(subdir, model);
					model_to_dir = os.path.join(dst_path, brand, model);
					if not os.path.exists(model_to_dir):
						os.makedirs(model_to_dir);
					total = len(os.listdir(model_from_dir))
					progress = ProgressBar(total=total, prefix=model, suffix='Done', decimals=3, length=50, fill='\u2588', zfill='-');
					for i, filename in enumerate(os.listdir(model_from_dir)):
						progress.print_progress_bar(i);
						try:
							img = Image.open(os.path.join(src_path, brand, model, filename));
							new_img = img.resize((128, 128));
							if not os.path.exists(model_to_dir):
								os.makedirs(model_to_dir)
							new_img.save(os.path.join(model_to_dir, filename));
						except e:
							continue
				print("\n");


def resize_image(img_name):
	try:
		img = Image.open(img_name);
		img = img.convert('RGB');
		return preprocess(img);
	except e:
		return None;


def images_to_array(img_dir_path, labels):
	"""
			Args:
					path (string): Path to images dir.
					labels ([string]): Image categories labels.
			Return ([([int, int], int)]): Data set of numpy image array and label index num.
	"""
	dataset = [];
	for model in labels:
		path = os.path.join(img_dir_path, model);
		model_index = labels.index(model);
		total = len(os.listdir(path))
		progress = ProgressBar(total=total, prefix="Preparing data", suffix=model, decimals=3, length=50, fill='\u2588', zfill='-');
		for i, img_name in enumerate(os.listdir(path)):
			progress.print_progress_bar(i + 1);
			try:
				img = Image.open(os.path.join(path, img_name));
			except e:
				continue;
			img = img.convert('RGB');
			img_tensor = preprocess(img);
			# service.img_show(img_tensor);
			# img_tensor = np.array(img); #cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE);
			if img_tensor is None:
				continue;
			dataset.append((img_tensor, model_index));
	random.shuffle(dataset);
	return dataset;


def image_tranform_to_tensor(image_name):
	img = Image.open(image_name);
	# img = resize_image(image_name);
	# if not img:
	# 	return None;
	# np_array = np.array(img);
	# img_array = cv2.cvtColor(np_array, cv2.IMREAD_GRAYSCALE);
	# to_tensor = transforms.ToTensor();
	return preprocess(img);


if __name__ == "__main__":
	from config import *;
	# resize_multiple_images(ORIG_IMG_DIR, IMG_DIR);
	images_to_array(os.path.join(ORIG_IMG_DIR, "Nike"), MODELS);
