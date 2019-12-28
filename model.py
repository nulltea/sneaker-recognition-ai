import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from console_progressbar import ProgressBar


class Net(nn.Module):
	def __init__(self):
		super().__init__();
		# Convolutional layers
		self.conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=6, kernel_size=6),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=6),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.ReLU(),
		);

		# Linear layers
		self.linear_layers = nn.Sequential(
			nn.Linear(32 * 11 * 11, 120),  # input
			nn.ReLU(),
			nn.Linear(120, 84),  # hidden
			nn.ReLU(),
			nn.Linear(84, 5),  # output
		);

	def forward(self, input):
		input = self.conv_layers(input);
		input = input.view(-1, 32 * 11 * 11);
		input = self.linear_layers(input);
		return input;
