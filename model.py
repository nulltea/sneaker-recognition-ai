import torchvision;
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import service;
from dataset import *;
from torch.utils.tensorboard import SummaryWriter;
from console_progressbar import ProgressBar;
from config import *;

class Net(nn.Module):
	def __init__(self, model_save_path = None):
		super(Net, self).__init__()
		self.model_save_path = model_save_path;

		#Convolutional layers
		self.conv1 = nn.Conv2d(3, 6, 6)
		self.conv2 = nn.Conv2d(6, 16, 6)
		self.conv3 = nn.Conv2d(16, 32, 6)
		#self.conv4 = nn.Conv2d(32, 64, 5)

		self.pool = nn.MaxPool2d(2, 2)

		#Layers
		self.fc1 = nn.Linear(32 * 11 * 11, 120)	#input
		self.fc2 = nn.Linear(120, 84)			#hidden
		self.fc3 = nn.Linear(84, 10)			#output
		
		#TensorBoard Writter
		self.tensorboard = SummaryWriter('runs/sneaker_net');

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)));
		x = self.pool(F.relu(self.conv2(x)));
		x = self.pool(F.relu(self.conv3(x)));
		s = x.size();
		x = x.view(-1, 32 * 11 * 11);
		x = F.relu(self.fc1(x));
		x = F.relu(self.fc2(x));
		x = self.fc3(x);
		return x;


	def train(self, data_set):
		data_loader = torch.utils.data.DataLoader(data_set, batch_size=2, num_workers=0, shuffle=True)
		#self.tensorboard.add_graph(self, data_loader);
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
		
		sub_epoch = 0;
		for epoch in range(20):  # loop over the dataset multiple times
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
			#log stats
			

			#log net state
			self.tensorboard.add_histogram("conv1.bias", self.conv1.bias, epoch);
			self.tensorboard.add_histogram("conv1.weight", self.conv1.weight, epoch);
			self.tensorboard.add_histogram("conv1.weight.grad", self.conv1.weight.grad, epoch);
		self.tensorboard.close();
		print('\nFinished Training')
		if self.model_save_path:
			torch.save(self.state_dict(), self.model_save_path) #save model

	def load_model(self):
		self.load_state_dict(torch.load(self.model_save_path))

	def __get_correct_total(self, preds, labels):
		return preds.argmax(dim=1).eq(labels).sum().item()

if __name__ == "__main__":
	dataset = SneakersDataset(IMG_DIR, MODELS);
	net = Net(MODEL_SAVE_PATH);
	net.load_model();
	print("Init training protocol? (y/n)");
	if input() == "y":
		net.train(data_set=dataset);