from PIL import Image;
from console_progressbar import ProgressBar;
import os;
import json;

class DataTransform():
	@staticmethod
	def resize_multiple_images(src_path, dst_path):
		# Here src_path is the location where images are saved.
		total = len(os.listdir(src_path))
		progress = ProgressBar(total=total, prefix="Resizing", suffix='Done', decimals=3, length=50, fill='X', zfill='-');
		index = 0;
		for i, filename in enumerate(os.listdir(src_path)):
			progress.print_progress_bar(i);
			try:
				img=Image.open(src_path+'\\'+filename)
				new_img = img.resize((128,128))
				if not os.path.exists(dst_path):
					os.makedirs(dst_path)
				new_img.save(dst_path+'\\'+filename)
				index += 1;
			except:
				continue
DataTransform.resize_multiple_images("Dataset\originals", "Dataset\images")
