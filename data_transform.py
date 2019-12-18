import os;
import cv2;
import json;
import random;
import numpy as np;
import matplotlib.pyplot as plt;
from PIL import Image;
from console_progressbar import ProgressBar;

def resize_multiple_images(src_path, dst_path):
	"""
		Args:
			src_path (string): location where images are saved.
			src_path (string): location where images will be saved.
	"""
	# Here src_path is the location where images are saved.
	total = len(os.listdir(src_path))
	progress = ProgressBar(total=total, prefix="Resizing", suffix='Done', decimals=3, length=50, fill='X', zfill='-');
	index = 0;
	for i, filename in enumerate(os.listdir(src_path)):
		progress.print_progress_bar(i);
		model_list = filename.split("_");
		model_dir = os.path.join(dst_path, model_list[0], model_list[1]);
		if not os.path.exists(model_dir):
			os.makedirs(model_dir);
		try:
			img=Image.open(os.path.join(src_path, filename));
			new_img = img.resize((128,128));
			if not os.path.exists(dst_path):
				os.makedirs(dst_path)
			new_img.save(os.path.join(model_dir, filename));
		except:
			continue


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
		for img in os.listdir(path):
			img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE);
			dataset.append((img_array, model_index));
	random.shuffle(dataset);
	return dataset;



