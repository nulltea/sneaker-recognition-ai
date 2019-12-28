from functools import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class Conv2dAuto(nn.Conv2d):
	"""
	Convolution layer class with a property to auto calc & add padding based on karnel size
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs);
		self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2);


# Partial functional object that represents auto-padding 3x3 convolutional layer
Conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False);
# functools.partial() - return a new partial object which when called will behave like func called with the positional arguments args and keyw


# ModuleDict to create a dictionary with different activation functions
def activation_func(activation):
	return nn.ModuleDict([
		['relu', nn.ReLU(inplace=True)],
		['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
		['selu', nn.SELU(inplace=True)],
		['none', nn.Identity()]
	])[activation];


class ResidualBlock(nn.Module):
	"""
	*Residual Block*
	The residual block takes an input with in_channels,
	applies some blocks of convolutional layers to reduce it to out_channels and sum it up to the original input.
	If their sizes mismatch, then the input goes into an identity.
	We can abstract this process and create a interface that can be extedend.
	"""
	def __init__(self, in_channels, out_channels, activation='relu'):
		super().__init__();
		self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation;
		self.blocks = nn.Identity();
		self.activate = activation_func(activation);
		self.shortcut = nn.Identity();

	def forward(self, x):
		residual = x;
		if self.should_apply_shortcut:
			residual = self.shortcut(x);
		x = self.blocks(x);
		x += residual;
		x = self.activate(x);
		return x;

	@property
	def should_apply_shortcut(self):
		return self.in_channels != self.out_channels;


class ResNetResidualBlock(ResidualBlock):
	"""
	In ResNet each block has a expansion parameter in order to increase the out_channels.
	Also, the identity is defined as a Convolution followed by an Activation layer, this is referred as shortcut.
	Then, we can just extend ResidualBlock and defined the shortcut function.
	"""
	def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=Conv3x3, *args, **kwargs):
		super().__init__(in_channels, out_channels);
		self.expansion, self.downsampling, self.conv = expansion, downsampling, conv;
		self.shortcut = nn.Sequential(
			nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
			nn.BatchNorm2d(self.expanded_channels)
		) if self.should_apply_shortcut else None;

	@property
	def expanded_channels(self):
		return self.out_channels * self.expansion;

	@property
	def should_apply_shortcut(self):
		return self.in_channels != self.expanded_channels;


# Function to stack one conv and batchnorm layer
def conv_batchnorm(in_channels, out_channels, conv, *args, **kwargs):
	return nn.Sequential(
		conv(in_channels, out_channels, *args, **kwargs),
		nn.BatchNorm2d(out_channels)
	);


class ResNetBasicBlock(ResNetResidualBlock):
	"""
	A basic ResNet block is composed by two layers of 3x3 convs/batchnorm/relu.
	"""
	expansion = 1

	def __init__(self, in_channels, out_channels, *args, **kwargs):
		super().__init__(in_channels, out_channels, *args, **kwargs)
		self.blocks = nn.Sequential(
			conv_batchnorm(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
			activation_func(self.activation),
			conv_batchnorm(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
		);


class ResNetBottleNeckBlock(ResNetResidualBlock):
	"""
	BottleNeck block is:
		The 3 layers are 1x1, 3x3, and 1x1 convolutions,
		where the 1x1 layers are responsible for reducing and then increasing (restoring) dimensions,
		leaving the 3x3 layer a bottleneck with smaller input/output dimensions.
	Used to increase the network deepths but to decrese the number of parameters.
	"""
	expansion = 4;

	def __init__(self, in_channels, out_channels, *args, **kwargs):
		super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs);
		self.blocks = nn.Sequential(
			conv_batchnorm(self.in_channels, self.out_channels, self.conv, kernel_size=1),
			activation_func(self.activation),
			conv_batchnorm(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
			activation_func(self.activation),
			conv_batchnorm(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
		);


class ResNetLayer(nn.Module):
	"""
	A ResNet's layer is composed by blocks stacked one after the other.
	it defines by stuck N blocks one after the other/
	** First convolution block has a stride of two,
	since "We perform downsampling directly by convolutional layers that have a stride of 2"
	"""
	def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
		super().__init__()
		downsampling = 2 if in_channels != out_channels else 1;
		self.blocks = nn.Sequential(
			block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
			*[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
		);

	def forward(self, x):
		x = self.blocks(x)
		return x
