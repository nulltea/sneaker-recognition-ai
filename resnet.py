import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from console_progressbar import ProgressBar
from torch.utils.tensorboard import SummaryWriter

import service
from config import *
from dataset import *
from resnet_serv import *


class ResNetEncoder(nn.Module):
	"""
	ResNet encoder composed by increasing different layers with increasing features.
	"""

	def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
				activation='relu', block=ResNetBasicBlock, *args, **kwargs):
		super().__init__();

		self.blocks_sizes = blocks_sizes;
		self.gate = nn.Sequential(
			nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(self.blocks_sizes[0]),
			activation_func(activation),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		);

		self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]));
		self.blocks = nn.ModuleList([
			ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, block=block, *args, **kwargs),
			*[ResNetLayer(in_channels * block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
			for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
		]);

	def forward(self, x):
		x = self.gate(x);
		for block in self.blocks:
			x = block(x);
		return x


class ResnetDecoder(nn.Module):
	"""
	This class represents the tail of ResNet. It performs a global pooling and maps the output to the
	correct class by using a fully connected layer.
	"""

	def __init__(self, in_features, n_classes):
		super().__init__()
		self.avg = nn.AdaptiveAvgPool2d((1, 1))
		self.decoder = nn.Linear(in_features, n_classes)

	def forward(self, x):
		x = self.avg(x)
		x = x.view(x.size(0), -1)
		x = self.decoder(x)
		return x


class ResNet(nn.Module):
	def __init__(self, in_channels, n_classes, *args, **kwargs):
		super().__init__();
		self.encoder = ResNetEncoder(in_channels, *args, **kwargs);
		self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes);

	def forward(self, x):
		x = self.encoder(x);
		x = self.decoder(x);
		return x;


def resnet18(in_channels, n_classes):
	return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])


def resnet34(in_channels, n_classes):
	return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])


def resnet50(in_channels, n_classes):
	return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])


def resnet101(in_channels, n_classes):
	return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])


def resnet152(in_channels, n_classes):
	return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])
