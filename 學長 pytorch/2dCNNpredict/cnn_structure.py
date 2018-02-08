import torch.nn as nn
import torch

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels = 1,
				out_channels = FILTER_NUM[fn_i],
				kernel_size = (KERNEL_SIZE[ks_i],vec),
				stride = 1,
				padding = 0),
			# nn.ReLU(),
			nn.Sigmoid(),
			nn.AdaptiveAvgPool2d(output_size=(1,vec)),
			nn.Dropout(p = 0.3)
		)
		self.fullconnect = nn.Linear(FILTER_NUM[fn_i]*1*vec,len(EMOTION))
		self.softmax = nn.LogSoftmax()

	def forward(self, x):
		x = self.conv(x)
		x = x.view(x.size(0),-1)
		x = self.fullconnect(x)
		output = self.softmax(x)
		return output, x

