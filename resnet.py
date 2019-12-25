
import torchvision;
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import service;
from dataset import *;
from torch.utils.tensorboard import SummaryWriter;
from console_progressbar import ProgressBar;
from config import *;

USE_GPU = False;
EPOCHS = 20;

class Conv2dAuto(nn.Conv2d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

def activation_func(activation):
	return  nn.ModuleDict([
		['relu', nn.ReLU(inplace=True)],
		['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
		['selu', nn.SELU(inplace=True)],
		['none', nn.Identity()]
	])[activation]

class Net(nn.Module):
	def __init__(self, model_save_path = None):
		super(Net, self).__init__()
		self.model_save_path = model_save_path;

		self.init_conv = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)

		#Convolutional layers
		self.conv_layer1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5),
			nn.BatchNorm2d(kernel_size=2, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.ReLU(),
		);

		#Linear layers
		self.linear_layers = nn.Sequential(
			nn.Linear(64 * 10 * 10, 120),		#input
			nn.ReLU(),
			nn.Linear(120, 84),					#hidden
			nn.ReLU(),
			nn.Linear(84, 7),					#output
		)
		#TensorBoard Writter
		self.tensorboard = SummaryWriter('runs/sneaker_net');

	def forward(self, input):
		input = self.conv_layers(input);
		input = input.view(-1, 64 * 10 * 10);
		input = self.linear_layers(input);
		return input;


	def train(self, data_set):
		data_loader = torch.utils.data.DataLoader(data_set, batch_size=4, num_workers=0, shuffle=True);
		
		if USE_GPU:
			self.cuda();

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.parameters(), lr=1e-3)
		
		sub_epoch = 0;
		for epoch in range(EPOCHS):
			running_loss = 0.0;
			total_loss = 0.0;
			running_correct = 0.0;
			total_correct = 0.0;
			train_list = enumerate(data_loader, 0);
			batch_size = 100;
			progress = ProgressBar(total=batch_size, prefix='Here', suffix='Now', decimals=3, length=50, fill='\u2588', zfill=' ')
			batch_index = 0;
			
			for i, data in train_list:
				progress.print_progress_bar(i - batch_index * batch_size);
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data;

				if USE_GPU:
					inputs = inputs.cuda();
					labels = labels.cuda();

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# statistics
				running_loss += loss.item();
				running_correct += self.__get_correct_total(outputs, labels);
				total_loss += loss.item();
				total_correct += self.__get_correct_total(outputs, labels);

				if i % batch_size == batch_size - 1:
					print('\t[%d, %5d] loss: %.3f correct: %5d' %(epoch + 1, i + 1, running_loss / batch_size, running_correct))
					self.tensorboard.add_scalar("Loss", running_loss, sub_epoch);
					self.tensorboard.add_scalar("Correct", running_correct, sub_epoch);
					self.tensorboard.add_scalar("Accuracy", running_correct / batch_size, sub_epoch);
					running_loss = 0.0;
					running_correct = 0.0;
					batch_index += 1;
					sub_epoch += 1;

		self.tensorboard.close();
		print('\nFinished Training')
		if self.model_save_path:
			torch.save(self.state_dict(), self.model_save_path) #save model

	def load_model(self):
		self.load_state_dict(torch.load(self.model_save_path))

	def __get_correct_total(self, preds, labels):
		return preds.argmax(dim=1).eq(labels).sum().item()

if __name__ == "__main__":
	dataset = SneakersDataset(os.path.join(IMG_DIR, "Nike"), MODELS);
	net = Net(MODEL_SAVE_PATH);
	#net.load_model();
	print("\nInit training protocol? (y/n)");
	if input() == "y":
		net.train(data_set=dataset);