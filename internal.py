import os;
import urllib.request;
from console_progressbar import ProgressBar;
from data_transform import *;

DIR = "Dataset/";

if __name__ == "__main__":
	for filename in os.listdir(DIR):
		if filename.endswith(".txt"):
			sneaker_model = os.path.splitext(filename)[0]
			with open("Dataset/{}".format(filename), 'r') as file:
				total = len(file.readlines());
				file.seek(0);
				url = file.readline()
				progress = ProgressBar(total=total, prefix=sneaker_model.ljust(15), suffix='Loaded', decimals=3, length=50, fill='X', zfill='-');
				i = 0;
				while url:
					url = file.readline();
					i += 1;
					progress.print_progress_bar(i);
					image = "Dataset/images/Nike_AirMax720_{}.jpg".format(i);
					if not os.path.exists(image):
						try: urllib.request.urlretrieve(url, "Dataset/originals/{0}_{1}.jpg".format(sneaker_model, i));
						except: continue;
	input();
	DataTransform.resize_multiple_images("Dataset\originals", "Dataset\images")