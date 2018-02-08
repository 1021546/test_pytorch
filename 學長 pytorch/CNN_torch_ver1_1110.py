import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import datetime
import scipy.io.wavfile
import numpy as np

start = datetime.datetime.now()

#hyper Parameters
EPOCH = 500
BATCH_SIZE = 50
LR = 0.02
INPUT_PATH = '../wav/'
EMOTION = {'ne':0, 'ha':1, 'sa':2, 'an':3, 'di':4, 'su':5, 'or':6, 'nx':7}
COV_KERNEL_SIZE = 512		#0.032*16000
COV_STEP_SIZE = 256			#0.016*16000
POOL_SIZE = 256
data_size = 1187
train_size = 900
test_size = data_size-train_size

x_data = []
y_data = []
file_list = os.listdir(INPUT_PATH)
for file in file_list:
	if file.endswith('.wav'):
		try:
			fstmp, wavtmp = scipy.io.wavfile.read(INPUT_PATH+file)
		except ValueError:
			print(file,' has unexpected end.')
			continue
		#wavtmp.dtype = int16
		nb_bits = 16
		samples = wavtmp / (float(2**(nb_bits - 1)) + 1.0)
		samples.reshape(samples.shape[0],1)
		x_data.append(samples)
		emo = file[-6:-4]
		y_data.append([EMOTION[emo]])
x_data = np.array(x_data)
y_data = np.array(y_data)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(  
			nn.Conv1d(
				in_channels=1,					#input dimension (1,1,length)
				out_channels=100,				#number of filters (1,200,length)
				kernel_size=COV_KERNEL_SIZE,	#filter size  
				stride=COV_STEP_SIZE,			#filter step size
				padding=COV_KERNEL_SIZE-COV_STEP_SIZE,	#padding for filtering
			),
			nn.ReLU(),
			nn.AdaptiveMaxPool1d(output_size=256),
			nn.Dropout(p=0.3)
		)
		self.conv2 = nn.Sequential(
			#nn.Conv1d(100,200,128,1,64),		#input(1,200,256)  output (1,100,256)
			nn.Conv1d(100,200,5,1,2),
			nn.ReLU(),
			nn.MaxPool1d(4),					#output(1,100,64)
			nn.Dropout(p=0.5)
		)
		self.fullconnect = nn.Linear(200*64,len(EMOTION))#100(conv2 output)*54(conv2input dim/MaxPool kernelsize), 4 for type of emotion
		#self.fullconnect = nn.Linear(100*256,len(EMOTION))
		self.softmax=nn.LogSoftmax()
		

	def forward(self,x):
		x = self.conv1(x)			#[1,200,256]
		x = self.conv2(x)			#[1,100,64]
		x = x.view(x.size(0),-1)	#size(0)=batch_size, -1 the output shape of MaxPool1d times together
		x = self.fullconnect(x)
		output = self.softmax(x)
		return output

cnn = CNN()
cnn.cuda().float()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()

index = np.arange(data_size)
np.random.shuffle(index)
x_data = x_data[index]
y_data = y_data[index]

test_x = x_data[-test_size:]
test_y = y_data[-test_size:]
train_x = x_data[:train_size]
train_y = y_data[:train_size]
# train_x = x_data[:] #inside
# train_y = y_data[:] #inside

for epoch in range(EPOCH):
	result=[]
	batchi=0
	lastloss = 0
	index = np.arange(train_size)	#outside
	# index = np.arange(data_size)	#indside
	np.random.shuffle(index)
	train_x = train_x[index]
	train_y = train_y[index]
	
	for i in range(train_size):
		x_tensor = Variable(torch.from_numpy(train_x[i][np.newaxis,np.newaxis,:])).cuda().float()
		output = cnn(x_tensor)
		result.append(output)
		if len(result)==BATCH_SIZE:
			result_va = torch.cat(result,0)
			y_va = Variable(torch.from_numpy(train_y[batchi*BATCH_SIZE:i+1]).view(BATCH_SIZE)).cuda()
			batchi+=1
			result=[]
			optimizer.zero_grad()
			loss = loss_func(result_va,y_va)
			loss.backward()
			optimizer.step()
			lastloss = loss.data[0]
	acc_count=0
	result_dis = [0,0,0,0,0,0,0,0]
	for i in range(test_size):
		x_test = Variable(torch.from_numpy(test_x[i][np.newaxis,np.newaxis,:])).cuda().float()
		test_out = cnn(x_test)
		pred_y = torch.max(test_out,1)[1].cuda().data.squeeze()
		result_dis[pred_y[0]]+=1
		if pred_y[0]==test_y[i]:
			acc_count +=1
	acc = acc_count/test_size
	
	print('neutral:',result_dis[0])
	print('sad:    ',result_dis[1])
	print('happy:  ',result_dis[2])
	print('anger:  ',result_dis[3])
	print('disgust:',result_dis[4])
	print('surprise:',result_dis[5])
	print('boredom:',result_dis[6])
	print('anxiety:',result_dis[7])
	
	print('Epoch:  ', epoch,'| train loss: %.6f' % lastloss, '| test accuracy: %.4f ' % acc)
	print()
print(cnn)
print('batch size:             ',BATCH_SIZE)
print('learning rate:          ',LR)
print('convolution kernel size:',COV_KERNEL_SIZE)
print('convolution step size:  ',COV_STEP_SIZE)
torch.save(cnn,'cnn2layer_4emo_batch'+str(BATCH_SIZE)+'.pt')
