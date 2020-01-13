import matplotlib.pyplot as plt;
import numpy as np;

def img_show(img):
	#img = img / 2 + 0.5		# unnormalize
	npimg = img.detach().numpy()
	s = img.size();
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()