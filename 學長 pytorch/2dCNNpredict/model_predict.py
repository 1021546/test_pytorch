import torch
import torch.nn
import os,csv,datetime
import numpy as np
from cnn_structure import CNN
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

# MODEL_FOLDER = './model/'
MODEL_FOLDER = './'
DATA_FOLDER = './k1000_vec200/'
EMOTION = {'ne':0, 'ha':1, 'sa':2, 'an':3, 'di':4, 'su':5, 'or':6, 'nx':7}


file_list = os.listdir(MODEL_FOLDER)
# cnn = []
# for file in file_list:
# 	if file.endswith('.pt'):
# 		cnn.append(torch.load(MODEL_FOLDER+file))
input_data = []
input_tag = []
file_list = os.listdir(DATA_FOLDER)
begin_read = datetime.datetime.now()

for file in file_list:
	if file.endswith('.csv'):
		frames = []
		csvreader = csv.reader(open(DATA_FOLDER+file))
		for frame in csvreader:
			tmp = np.array(frame)
			tmp = [float(i) for i in tmp]
			frames.append(tmp)
		input_data.append(frames)
		input_tag.append(EMOTION[file[-16:-14]])
input_data = np.array(input_data)
input_tag = np.array(input_tag)
end_time = datetime.datetime.now()
print('time of read ',len(input_data),' files: ',datetime.datetime.now()-begin_read)

cnn = torch.load('./k_1000_vec200_featureCNN_8emo_fold0date1227.pt')



acc_count = 0
results = []
time = []
for i in range(len(input_data)):
	begin = datetime.datetime.now()
	data_tensor = Variable(torch.from_numpy(np.array([[input_data[i]]]))).cuda().float()
	# tag_tensor = Variable(torch.from_numpy([test_tag[i]])).cuda().float()
	test_output, lastlayer = cnn(data_tensor)
	tag_output = torch.max(test_output,1)[1].cuda().data.squeeze()
	# data_tensor = Variable(torch.from_numpy(np.array([[test_data[i]]]))).float()
	# test_output, lastlayer = cnn(data_tensor)
	# tag_output = torch.max(test_output,1)[1].data.squeeze()
	print(tag_output[0], ' time:',datetime.datetime.now()-begin)
	time.append(datetime.datetime.now()-begin)

	results.append(tag_output[0])
	if tag_output[0]==input_tag[i]:
		acc_count+=1
	acc = acc_count/(len(input_tag))
confusion = confusion_matrix(input_tag,results)
print(acc)
print(confusion)
# print('avg_time:',sum(time)/len(time))