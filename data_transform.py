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
	for subdir, brands, _ in os.walk(src_path):
		for brand in brands:
			for subdir, models, _ in os.walk(os.path.join(subdir, brand)):
				for model in models:
					model_from_dir = os.path.join(subdir, model);
					model_to_dir = os.path.join(dst_path, brand, model);
					if not os.path.exists(model_to_dir):
						os.makedirs(model_to_dir);
					total = len(os.listdir(model_from_dir))
					progress = ProgressBar(total=total, prefix=model, suffix='Done', decimals=3, length=50, fill='X', zfill='-');
					for i, filename in enumerate(os.listdir(model_from_dir)):
						progress.print_progress_bar(i);
						
						try:
							img=Image.open(os.path.join(src_path, brand, model, filename));
							new_img = img.resize((128,128));
							if not os.path.exists(model_to_dir):
								os.makedirs(model_to_dir)
							new_img.save(os.path.join(model_to_dir, filename));
						except:
							continue
				print("\n");


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
			if img_array is None:
				continue;
			dataset.append((img_array, model_index));
	random.shuffle(dataset);
	return dataset;



if __name__ == "__main__":
	from config import *;
	resize_multiple_images(ORIG_IMG_DIR, IMG_DIR)