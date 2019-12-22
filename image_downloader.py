import os;
import urllib.request;
import requests;
import mimetypes;
from console_progressbar import ProgressBar;
from data_transform import *;
from config import *;

def download_images():
	for filename in os.listdir(DATA_DIR):
		if filename.endswith(".txt"):
			sneaker_model_names = os.path.splitext(filename)[0].split('_');
			sneaker_brand = sneaker_model_names[0];
			sneaker_model = sneaker_model_names[1];
			model_dir = os.path.join(ORIG_IMG_DIR, sneaker_brand, sneaker_model);
			with open(os.path.join(DATA_DIR, filename), 'r') as file:
				total = len(file.readlines());
				file.seek(0);
				url = file.readline()
				progress = ProgressBar(total=total, prefix=sneaker_brand, suffix=sneaker_model, decimals=3, length=50, fill='\u2588', zfill='-');
				i = 0;
				while url:
					url = file.readline();
					image_path = os.path.join(model_dir, "{0}_{1}_{2}.jpg".format(sneaker_brand, sneaker_model, i));
					if not os.path.exists(model_dir):
						os.makedirs(model_dir);
					if not os.path.exists(image_path):
						try: urllib.request.urlretrieve(url, image_path);
						except: continue;
					i += 1;
					progress.print_progress_bar(i);
					if i == 300:
						break;
		print("\n");

def get_url_extension(url):
	response = requests.get(url)
	content_type = response.headers['content-type']
	return mimetypes.guess_extension(content_type)

if __name__ == "__main__":
	download_images();
