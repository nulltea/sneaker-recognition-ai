import torch.distributions.lowrank_multivariate_normal;
import torchvision;
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import *;
from torch.utils.tensorboard import SummaryWriter;
from console_progressbar import ProgressBar;
from config import *;

class Net(nn.Module):
	def __init__(self, model_save_path):
		super(Net, self).__init__()
		self.model_save_path = model_save_path;

		#Convolutional layers
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)

		self.pool = nn.MaxPool2d(2, 2)

		#Layers
		self.fc1 = nn.Linear(16 * 29 * 29, 120)	#input
		self.fc2 = nn.Linear(120, 84)			#hidden
		self.fc3 = nn.Linear(84, 10)			#output

		#TensorBoard Writter
		self.tensorboard = SummaryWriter('runs/sneaker_net_test');



	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)));
		x = self.pool(F.relu(self.conv2(x)));
		x = x.view(-1, 16 * 29 * 29);
		x = F.relu(self.fc1(x));
		x = F.relu(self.fc2(x));
		x = self.fc3(x);
		return x;

	def train(self, data_set):
		data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, num_workers=0, shuffle=True, pin_memory=False)
		self.tensorboard.add_graph(self, data_loader);
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
		for epoch in range(2):  # loop over the dataset multiple times
			running_loss = 0.0;
			total_loss = 0.0;
			train_list = enumerate(data_loader, 0);
			progress = ProgressBar(total=100, prefix='Here', suffix='Now', decimals=3, length=50, fill='X', zfill='-')
			epoch_index = 0;
			for i, data in train_list:
				progress.print_progress_bar(i - epoch_index * 100);
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data;

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				total_loss += loss.item();
				if i % 100 == 99:    # print every 2000 mini-batches
					print('\t[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
					running_loss = 0.0
					epoch_index += 1;
			self.tensorboard.add_scalar("Loss", total_loss, epoch);

			self.tensorboard.add_histogram("conv1.bias", self.conv1.bias, epoch);
			self.tensorboard.add_histogram("conv1.weight", self.conv1.weight, epoch);
			self.tensorboard.add_histogram("conv1.weight.grad", self.conv1.weight.grad, epoch);
		self.tensorboard.close();
		print('\nFinished Training')
		torch.save(self.state_dict(), self.model_save_path) #save model

	def load_model(self):
		self.load_state_dict(torch.load(self.model_save_path))