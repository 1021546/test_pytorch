import gzip
import os
from os import path
import urllib
import numpy as np
import glob
# import cv2
import random

DATASET_DIR = 'datasets/'

def load_mnist(ntrain=60000, ntest=10000):
	data_dir = os.path.join(DATASET_DIR,'mnist/')

	with gzip.open(os.path.join(data_dir,'train-images-idx3-ubyte.gz')) as fd:
		buf = fd.read()
		loaded = np.frombuffer(buf,dtype=np.uint8)
		trX = loaded[16:].reshape((60000,28*28)).astype(float)
	with gzip.open(os.path.join(data_dir,'train-labels-idx1-ubyte.gz')) as fd:
		buf = fd.read()
		loaded = np.frombuffer(buf,dtype=np.uint8)
		trY = loaded[8:].reshape((60000))
	with gzip.open(os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')) as fd:
		buf = fd.read()
		loaded = np.frombuffer(buf,dtype=np.uint8)
		teX = loaded[16:].reshape((10000,28*28)).astype(float)
	with gzip.open(os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')) as fd:
		buf = fd.read()
		loaded = np.frombuffer(buf,dtype=np.uint8)
		teY = loaded[8:].reshape((10000))

	trX /= 255.
	teX /= 255.
	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	trY = np.asarray(trY)
	teY = np.asarray(teY)
	return trX, teX, trY, teY

import numpy as np
import torch
from torch.autograd import Variable
from torch import optim

class ConvNet(torch.nn.Module):
	def __init__(self,output_dim):
		super(ConvNet,self).__init__()

		self.conv = torch.nn.Sequential()
		self.conv.add_module("conv_1",torch.nn.Conv2d(1, 10, kernel_size=5))
		self.conv.add_module("maxpool_1",torch.nn.MaxPool2d(kernel_size=2))
		self.conv.add_module("relu_1",torch.nn.ReLU())
		self.conv.add_module("conv_2",torch.nn.Conv2d(10, 20, kernel_size=5))
		self.conv.add_module("dropout_2",torch.nn.Dropout())
		self.conv.add_module("maxpool_2",torch.nn.MaxPool2d(kernel_size=2))
		self.conv.add_module("relu_2",torch.nn.ReLU())

		self.fc = torch.nn.Sequential()
		self.fc.add_module("fc1",torch.nn.Linear(20*4*4,50))
		self.conv.add_module("relu_3",torch.nn.ReLU())
		self.conv.add_module("dropout_3",torch.nn.Dropout())
		self.fc.add_module("fc2",torch.nn.Linear(50,output_dim))

	def forward(self,x):
		x = self.conv.forward(x)
		x = x.view(-1,20*4*4)
		return self.fc.forward(x)

def train(model, loss, optimizer, x_val, y_val):
	x = Variable(x_val.cuda(),requires_grad=False)
	y = Variable(y_val.cuda(),requires_grad=False)

	optimizer.zero_grad()

	fx = model.forward(x)
	output = loss.forward(fx,y)

	output.backward()

	optimizer.step()

	# print(output)
	# print(output.data[0])
	# input()

	# Variable containing:
 	# 	2.3205
	# [torch.cuda.FloatTensor of size 1 (GPU 0)]

	# 2.3204822540283203

	# 	Variable containing:
	#  2.3281
	# [torch.FloatTensor of size 1]

	# 2.32810378074646

	return output.data[0]

def predict(model, x_val):
	x = Variable(x_val.cuda(), requires_grad=False)
	output = model.forward(x)
	return output.cpu().data.numpy().argmax(axis=1)

def main():
	torch.manual_seed(42)
	trX, teX, trY, teY = load_mnist()
	# print(trX.shape)
	# print(teX.shape)
	# print(trY.shape)
	# print(teY.shape)
	# input()

	# (60000, 784)
	# (10000, 784)
	# (60000,)
	# (10000,)



	trX = trX.reshape(-1, 1, 28, 28)
	teX = teX.reshape(-1, 1, 28, 28)

	# (60000, 1, 28, 28)
	# (10000, 1, 28, 28)
	# (60000,)
	# (10000,)

	# print(trX.shape)
	# print(teX.shape)
	# print(trY.shape)
	# print(teY.shape)
	# input()

	trX = torch.from_numpy(trX).float()
	teX = torch.from_numpy(teX).float()

	trY = torch.from_numpy(trY).long()
	# teY = torch.from_numpy(teY).long()

	n_examples = len(trX)

	# print(trX.shape)
	# print(teX.shape)
	# print(trY.shape)
	# print(teY.shape)

	# print(n_examples)
	# input()

	# torch.Size([60000, 1, 28, 28])
	# torch.Size([10000, 1, 28, 28])
	# torch.Size([60000])
	# torch.Size([10000])
	# 60000

	n_classes = 10
	model = ConvNet(output_dim=n_classes)
	model.cuda(0)


	loss = torch.nn.CrossEntropyLoss(size_average=True)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	batch_size = 100
	epochs = 10

	for i in range(epochs):
		cost = 0.
		num_batches = n_examples / batch_size
		for k in range(int(num_batches)):
			start, end = k*batch_size, (k+1)*batch_size
			# print(trX[start:end].shape)
			# print(trY[start:end].shape)
			# input()
			# torch.Size([100, 1, 28, 28])
			# torch.Size([100])
			cost += train(model, loss, optimizer, trX[start:end], trY[start:end])
		predY = predict(model, teX)
		print("Epoch %d, cost = %f, acc = %.2f%%" % (i+1, cost/num_batches, 100.*np.mean(predY == teY)))

if __name__ == '__main__':
	main()