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

x_train=np.empty(shape=[0, 99, 39])

for k in range(1,7):
	for i in range(0,6):
		for j in range(1,6):
			one_train=np.empty(shape=[0, 39])
			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_"+str(j)+".wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			one_train=np.vstack((one_train,result_1[0:99]))

			# print(one_train)
			# print(one_train.shape)
			# print(type(one_train))

			one_train = np.reshape(one_train, (1, 99, 39))

			x_train=np.vstack((x_train,one_train))

			# print(x_train)
			# print(x_train.shape)
			# print(type(x_train))




# print(x_train)
# print(x_train.shape)
# print(type(x_train))

# print(x_train[0])
# print(x_train[0].shape)
# print(x_train.shape[0])
# print(x_train.shape[1])



x_test=np.empty(shape=[0, 99, 39])

for k in range(1,7):
	for i in range(0,6):
			one_test=np.empty(shape=[0, 39])
			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_4.wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			one_test=np.vstack((one_test,result_1[0:99]))

			one_test = np.reshape(one_test, (1, 99, 39))

			x_test=np.vstack((x_test,one_test))

# print(x_test)
# print(x_test.shape)
# print(type(x_test))


y_train=np.zeros(180, dtype=np.int)

index=0
for i in range(0,6):
	y_train[index:(index+30)]=i
	index+=30

print(y_train)
print(y_train.shape)
print(type(y_train))

y_test=np.zeros(36, dtype=np.int)

index=0
for i in range(0,6):
	y_test[index:(index+6)]=i
	index+=6

print(y_test)
print(y_test.shape)
print(type(y_test))

data=torch.from_numpy(x_train)
label=torch.from_numpy(y_train)

BATCH_SIZE = 5      # 批训练的数据个数
EPOCH = 2
# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(data_tensor=data, target_tensor=label)

# 把 dataset 放入 DataLoader
train_loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=2,              # 多线程来读数据
)

test_x = Variable(torch.from_numpy(x_test)).float().cuda()
test_y = Variable(torch.from_numpy(y_test)).long().cuda()
print(test_x.size())
test_x = test_x.unsqueeze(1)
print(test_x.size())

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 99, 39)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 99, 39)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 49, 19)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 49, 19)
            nn.Conv2d(
            	in_channels=16,             # input height
                out_channels=32,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),     							# output shape (32, 49, 19)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # output shape (32, 24, 9)
        )
        self.out = nn.Linear(32 * 24 * 9, 6)   # fully connected layer, output 6 classes
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        output = self.softmax(output)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture
cnn.cuda()


LR = 0.001              # learning rate

# print(cnn.parameters())

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# first version
# b_x = Variable(torch.from_numpy(x_train[0])).float()
# b_y = Variable(torch.from_numpy(y_train)).long()

# b_x = b_x.unsqueeze(0)
# b_x = b_x.unsqueeze(1)							# reshape it such that it has a batch dimension
# print(b_x.size())

# # print(b_x[0].size())

# output = cnn(b_x)[0]                             	# cnn output
# print(output)
# print(b_y[0])
# loss = loss_func(output, b_y[0])                # cross entropy loss
# optimizer.zero_grad()                           # clear gradients for this training step
# loss.backward()                                 # backpropagation, compute gradients
# optimizer.step()								# apply gradients

# second version
# b_x = Variable(torch.from_numpy(x_train)).float()
# b_y = Variable(torch.from_numpy(y_train)).long()


# b_x = b_x.unsqueeze(1)							# reshape it such that it has a batch dimension
# print(b_x.size())

# # print(b_x[0].size())

# output = cnn(b_x)[0]                             	# cnn output
# print(output)
# print(b_y)
# loss = loss_func(output, b_y)                # cross entropy loss
# optimizer.zero_grad()                           # clear gradients for this training step
# loss.backward()                                 # backpropagation, compute gradients
# optimizer.step()								# apply gradients


# third version
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x).float().cuda()  # batch x
        b_y = Variable(y).long().cuda()   # batch y

        b_x = b_x.unsqueeze(1)
        # print(b_x.size())
        # torch.Size([5, 1, 99, 39]) [batch_size, channel, 99, 39]

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        # print(output)
        # print(b_y)
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 10 == 0:
        	test_output, last_layer = cnn(test_x)
        	pred_y = torch.max(test_output, 1)[1]
        	# print(pred_y)
        	# print(test_y)
        	accuracy = sum(pred_y == test_y) / float(test_y.size(0))
        	print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)