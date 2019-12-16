from torch.utils.data import Dataset;
from PIL import Image;
import os;
import numpy as np;
import matplotlib.pyplot as plt;
import cv2;


MODELS = ['AirMax720', 'AirMax720', 'AirMax95', 'AirForce1', 'Jordan', 'React', 'VaporMax'];


DIR = "Dataset\\images\\Nike"

class DataSetCreator:
	def create(self):
		dataset = [];
		for model in MODELS:
			path = os.path.join(DIR, model);
			model_index = MODELS.index(model);
			for img in os.listdir(path):
				img_array = cv2.imread(os.path.join(DIR, model, img), cv2.IMREAD_GRAYSCALE);
				print(img_array)
				plt.imshow(img_array, cmap="gray");
				plt.show();
				dataset.append([img_array, model_index]);
	return dataset;

dataset_creator = DataSetCreator();
dataset_creator.create();
input()