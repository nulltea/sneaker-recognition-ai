import os;
import urllib.request;
from console_progressbar import ProgressBar;
from data_transform import *;
from config import *;

def download_images():
	for filename in os.listdir(DATA_DIR):
		if filename.endswith(".txt"):
			sneaker_model_names = os.path.splitext(filename)[0].split('_');
			sneaker_brand = sneaker_model_names[0];
			sneaker_model = sneaker_model_names[1];
			with open("Dataset/{}".format(filename), 'r') as file:
				total = len(file.readlines());
				file.seek(0);
				url = file.readline()
				progress = ProgressBar(total=total, prefix=sneaker_brand, suffix=sneaker_model, decimals=3, length=50, fill='X', zfill='-');
				i = 0;
				while url:
					url = file.readline();
					i += 1;
					progress.print_progress_bar(i);
					image_path = os.path.join(ORIG_IMG_DIR, "{0}_{1}_{2}.jpeg".format(sneaker_brand, sneaker_model, i));
					if not os.path.exists(image_path):
						try: urllib.request.urlretrieve(url, image_path);
						except: continue;

if __name__ == "__main__":
	download_images();
	input();
	resize_multiple_images(ORIG_IMG_DIR, IMG_DIR)