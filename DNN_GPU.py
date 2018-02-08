import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import torch.utils.data as Data

x_train=np.empty(shape=[0, 39])

for k in range(1,7):
	for i in range(0,6):
		for j in range(1,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_"+str(j)+".wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			x_train=np.vstack((x_train,result_1[0:99]))




# print(x_train)
# print(x_train.shape)
# print(type(x_train))

# print(x_train[0])
# print(x_train[0].shape)
# print(x_train.shape[0])
# print(x_train.shape[1])


x_test=np.empty(shape=[0, 39])


for k in range(1,7):
	for i in range(0,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_4.wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			x_test=np.vstack((x_test,result_1[0:99]))

# print(x_test)
# print(x_test.shape)
# print(type(x_test))


y_train=np.zeros(17820, dtype=np.int)

index=0
for i in range(0,6):
	y_train[index:(index+2970)]=i
	index+=2970

# y_train[0:2970]=0
# y_train[2970:5940]=1
# y_train[5940:8910]=2
# y_train[8910:11880]=3
# y_train[11880:14850]=4
# y_train[14850:17820]=5

y_test=np.zeros(3564, dtype=np.int)

index=0
for i in range(0,6):
	y_test[index:(index+594)]=i
	index+=594

# y_test[0:594]=0
# y_test[594:1188]=1
# y_test[1188:1782]=2
# y_test[1782:2376]=3
# y_test[2376:2970]=4
# y_test[2970:3564]=5

# train_loader = Data.DataLoader(x_train,batch_size=64,shuffle=True, drop_last=False)
# print(train_loader)
# for step, x in enumerate(train_loader):
# 	print(step)
# 	print(x)
# print(len(train_loader))
# train_loader_1 = Data.DataLoader(y_train,batch_size=64,shuffle=True, drop_last=False)
# print(train_loader_1)
# for step, x in enumerate(train_loader_1):
# 	print(step)
# 	print(x)
# print(len(train_loader_1))

# class DNN(nn.Module):
# 	def __init__(self):
# 		super(DNN, self).__init__()
# 		self.fc1 = nn.Linear(39, 50)
# 		# self.dropout = nn.Dropout(0.5),  # drop 50% of the neuron
# 		self.fc2 = nn.Linear(50, 128)
# 		self.out = nn.Linear(128, 10)

# 	def forward(self, x):
# 		x = self.fc1(x)
# 		x = F.relu(x)
# 		# x = self.dropout(x)
# 		x = F.dropout(x)
# 		x = self.fc2(x)
# 		x = F.relu(x)
# 		# x = self.dropout(x)
# 		x = F.dropout(x)
# 		x = self.out(x)
# 		return x

class DNN(nn.Module):
	def __init__(self):
		super(DNN, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(39, 50),
			nn.ReLU(),		# activation
			nn.Dropout(0.5),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(50, 128),
			nn.ReLU(),		# activation
			nn.Dropout(0.5),
		)
		self.out = nn.Linear(128, 10)
		self.softmax=nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.out(x)
		x = self.softmax(x)
		return x

net = DNN()
print(net)
net.cuda()

LR = 0.001              # learning rate

optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted


b_x = Variable(torch.from_numpy(x_train)).float().cuda()
b_y = Variable(torch.from_numpy(y_train)).long().cuda()

output = net(b_x)                             	# dnn output
# print(b_x.size())
loss = loss_func(output, b_y)                   # cross entropy loss
optimizer.zero_grad()                           # clear gradients for this training step
loss.backward()                                 # backpropagation, compute gradients
optimizer.step()								# apply gradients
x = Variable(torch.from_numpy(x_test)).float().cuda()
# print(x.size())
y = Variable(torch.from_numpy(y_test)).long().cuda()
test_output = net(x)
# print(test_output)
# print("Before: ",y)
pred_y = torch.max(test_output, 1)[1]
# print("After: ",pred_y)
# print(y.size(0))
accuracy = float(sum(pred_y == y)) / float(y.size(0))
print(accuracy)

# pred_y = torch.max(test_output, 1)[1]
# print("After: ",pred_y)
# accuracy = float(sum(pred_y == y)) / y.size(0)
# print(y.size(0))
# print(accuracy)


# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print("After: ",pred_y)
# accuracy = float(sum(pred_y == y_test)) / float(y.size(0))
# print(y.size(0))
# print(accuracy)


# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print("After: ",pred_y)
# accuracy = sum(pred_y == y_test) / float(y.size(0))
# print(y.size(0))
# print(accuracy)