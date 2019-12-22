from config import *;
from model import *;
from data_transform import *;
import tempfile;
import service;

def test_model(model, test_dir, models = MODELS, batch_size = 4):
	trainset = [];
	for image in os.listdir(test_dir):
		img_array = image_tranform_to_tensor(os.path.join(test_dir, image));
		if img_array is None:
			continue;
		trainset.append(img_array);
	dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True)
	dataiter = iter(dataloader)
	data_batch = dataiter.next();
	service.img_show(torchvision.utils.make_grid(data_batch))
	outputs = model(data_batch);
	_, predicted = torch.max(outputs, 1);
	print('Predicted: ', ', '.join('%5s' % models[predicted[j]] for j in range(batch_size)))

def img_show(img):
	#img = img / 2 + 0.5		# unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

activation = {}

def normalize_output(img):
	img = img - img.min()
	img = img / img.max()
	return img

def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

if __name__ == "__main__":
	net = Net(MODEL_SAVE_PATH);
	net.load_model();
	net.conv1.register_forward_hook(get_activation('conv2'))
	#net.conv1.register_forward_hook(get_activation('conv2'))
	#net.conv1.register_forward_hook(get_activation('conv3'))
	test_model(net, TEST_DIR, batch_size=4);

	j = 0
	f,axarr  = plt.subplots(4, 3)
	for layer in range(4):
		convx = "conv2"#.format(layer + 1)
		act = activation[convx].squeeze()
		print(act.size());
		for img in range(3):
			axarr[layer,img].imshow(act[layer][img]);
	plt.show()