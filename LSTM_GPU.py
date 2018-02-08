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
EPOCH = 1
TIME_STEP = 99          # rnn time step / image height
INPUT_SIZE = 39         # rnn input size / image width
LR = 0.01               # learning rate

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

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 6)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out, r_out, h_n, h_c


rnn = RNN()
print(rnn)
rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x).float().cuda()  # batch x
        b_y = Variable(y).long().cuda()   # batch y

        
        # print(b_x.size())
        # torch.Size([5, 99, 39]) [batch_size, time_step, input_size]

        # output, r_output, h_net, h_cet = rnn(b_x)               # rnn output
        # print(output)
        # input()
        # print(r_output)
        # input()
        # print(h_net)
        # input()
        # print(h_cet)
        # input()

        output = rnn(b_x)[0]            # rnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        # print(output)
        # print(b_y)
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 10 == 0:
        	test_output = rnn(test_x)[0]
        	pred_y = torch.max(test_output, 1)[1]
        	# print(pred_y)
        	# print(test_y)
        	accuracy = sum(pred_y == test_y) / float(test_y.size(0))
        	print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)