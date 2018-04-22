# import torch
# import numpy as np

# # Generate a 2-D pytorch tensor (i.e., a matrix)
# pytorch_tensor = torch.Tensor(10, 20)
# # print("type: ", type(pytorch_tensor), " and size: ", pytorch_tensor.shape )

# # Convert the pytorch tensor to a numpy array:
# numpy_tensor = pytorch_tensor.numpy()
# # print("type: ", type(numpy_tensor), " and size: ", numpy_tensor.shape)

# # Convert the numpy array to Pytorch Tensor:
# # print("type: ", type(torch.Tensor(numpy_tensor)), " and size: ", 
# # 	torch.Tensor(numpy_tensor).shape)

# # t = torch.rand(2, 4, 3, 5)
# # print(t)
# # a = np.random.rand(2, 4, 3, 5)
# # print(a)

# t = torch.rand(2, 4, 3, 5)
# a = t.numpy()
# pytorch_slice = t[0, 1:3, :, 4]
# numpy_slice =  a[0, 1:3, :, 4]
# # print ('Tensor[0, 1:3, :, 4]:\n', pytorch_slice)
# # print ('NdArray[0, 1:3, :, 4]:\n', numpy_slice)

# t = t - 0.5
# a = t.numpy()
# pytorch_masked = t[t > 0]
# numpy_masked = a[a > 0]
# # print ('pytorch_masked:\n', pytorch_masked)
# # print ('numpy_masked:\n', numpy_masked)

# pytorch_reshape = t.view([6, 5, 4])
# numpy_reshape = a.reshape([6, 5, 4])

# # print ('pytorch_reshape:\n', pytorch_reshape)
# # print ('numpy_reshape:\n', numpy_reshape)



# from torch.autograd import Variable
# import torch.nn.functional as F

# x = Variable(torch.randn(4, 1), requires_grad=False)
# y = Variable(torch.randn(3, 1), requires_grad=False)

# w1 = Variable(torch.randn(5, 4), requires_grad=True)
# w2 = Variable(torch.randn(3, 5), requires_grad=True)

# # `@` mean matrix multiplication from python3.5, PEP-0465
# def model_forward(x):
#     return F.sigmoid(w2 @ F.sigmoid(w1 @ x))

# # print (w1)
# # print (w1.data.shape)
# # print (w1.grad) # Initially, non-existent

# import torch.nn as nn
# criterion = nn.MSELoss()

# import torch.optim as optim
# optimizer = optim.SGD([w1, w2], lr=0.001)

# for epoch in range(10):
#     loss = criterion(model_forward(x), y)
#     optimizer.zero_grad() # Zero-out previous gradients
#     loss.backward() # Compute new gradients
#     optimizer.step() # Apply these gradients

# # print (w1)

# cuda_gpu = torch.cuda.is_available()
# if (cuda_gpu):
#     print("Great, you have a GPU!")
# else:
#     print("Life is short -- consider a GPU!")

# if cuda_gpu:
#     x = x.cuda()
#     print(type(x.data))

# x = x.cpu()
# print(type(x.data))

#--------------------------------------------------------------------------------

# from sklearn.datasets import make_regression
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt


# sns.set()

# x_train, y_train, W_target = make_regression(n_samples=100, n_features=1, noise=10, coef = True)

# df = pd.DataFrame(data = {'X':x_train.ravel(), 'Y':y_train.ravel()})

# sns.lmplot(x='X', y='Y', data=df, fit_reg=True)
# plt.show()

# x_torch = torch.FloatTensor(x_train)
# y_torch = torch.FloatTensor(y_train)
# y_torch = y_torch.view(y_torch.size()[0], 1)

# class LinearRegression(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_size, output_size)  

#     def forward(self, x):
#         return self.linear(x)

# model = LinearRegression(1, 1)

# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  


# for epoch in range(50):
#     data, target = Variable(x_torch), Variable(y_torch)
#     output = model(data)

#     optimizer.zero_grad()
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()

# predicted = model(Variable(x_torch)).data.numpy()

# plt.plot(x_train, y_train, 'o', label='Original data')
# plt.plot(x_train, predicted, label='Fitted line')
# plt.legend()
# plt.show()

# -----------------------------------------------------------------------

from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train(model, epoch, criterion, optimizer, data_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
            model.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 400 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(data_loader.dataset),
                100. * (batch_idx+1) / len(data_loader), loss.data[0]))


def test(model, epoch, criterion, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
            model.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader) # loss function already averages over batch size
    acc = correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * acc))
    return (acc, test_loss)


batch_num_size = 64

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), 
    batch_size=batch_num_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), 
    batch_size=batch_num_size, shuffle=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


cuda_gpu = torch.cuda.is_available()
if (cuda_gpu):
    print("Great, you have a GPU!")
else:
    print("Life is short -- consider a GPU!")

model = LeNet()
if cuda_gpu:
    model.cuda()

print ('MNIST_net model:\n')
print (model)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(),lr = 0.005, momentum = 0.9)


import os 

epochs = 5
if (os.path.isfile('pretrained/MNIST_net.t7')):
    print ('Loading model')
    model.load_state_dict(torch.load('pretrained/MNIST_net.t7', map_location=lambda storage, loc: storage))
    acc, loss = test(model, 1, criterion, test_loader)
else:
    print ('Training model')
    for epoch in range(1, epochs + 1):
        train(model, epoch, criterion, optimizer, train_loader)
        acc, loss = test(model, 1, criterion, test_loader)
    torch.save(model.state_dict(), 'pretrained/MNIST_net.t7')

print ('Internal models:')
for idx, m in enumerate(model.named_modules()):
    print(idx, '->', m)
    print ('-------------------------------------------------------------------------')

data = model.conv1.weight.cpu().data.numpy()
print (data.shape)
print (data[:, 0].shape)

kernel_num = data.shape[0]

fig, axes = plt.subplots(ncols=kernel_num, figsize=(2*kernel_num, 2))

for col in range(kernel_num):
    axes[col].imshow(data[col, 0, :, :], cmap=plt.cm.gray)
plt.show()


#---------------------------------------------------------------------

# import torch
# from torch.autograd import Variable

# t = Variable(torch.rand(2, 4, 3, 5), requires_grad=False)

# print (type(t.cpu().data))
# if torch.cuda.is_available():
#     print ("Cuda is available")
#     print (type(t.cuda().data))
# else:
#     print ("Cuda is NOT available")


# if torch.cuda.is_available():
#     try:
#         print(t.data.numpy())
#     except RuntimeError as e:
#         "you can't transform a GPU tensor to a numpy nd array, you have to copy your weight tendor to cpu and then get the numpy array"

# print(type(t.cpu().data.numpy()))
# print(t.cpu().data.numpy().shape)
# print(t.cpu().data.numpy())

