import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './mnist'
download = False  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)

# print(train_set)
# <torchvision.datasets.mnist.MNIST object at 0x000001B6321B9CF8>

test_set = dset.MNIST(root=root, train=False, transform=trans)
# print(test_set)
# <torchvision.datasets.mnist.MNIST object at 0x00000181018A0E48>

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


## network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        i = 0

    def forward(self, x): # 28x28
        x = F.relu(self.conv1(x)) # 24x24
        x = F.max_pool2d(x, 2, 2) # 12x12
        x = F.relu(self.conv2(x)) # 8x8
        x = F.max_pool2d(x, 2, 2) # 4x4
        # print(x.size()) # torch.Size([100, 50, 4, 4])
        x = x.view(-1, 4*4*50)
        # print(x.size()) # torch.Size([100, 800])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"

## training
model = LeNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

ceriation = nn.CrossEntropyLoss()

for epoch in range(10):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
        	x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        # print(batch_idx)
        out = model(x)
        # input()
        loss = ceriation(out, target)
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))

    print(batch_idx)
    
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
        	x, target = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        # print(type(out))
        # print(type(target))
        # <class 'torch.autograd.variable.Variable'>
		# <class 'torch.autograd.variable.Variable'>
        # input()
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))

    print(batch_idx)


torch.save(model.state_dict(), model.name())

# epoch = 1
# ==>>> total trainning batch number: 600
# ==>>> total testing batch number: 100
# ==>>> epoch: 0, batch index: 100, train loss: 0.719068
# ==>>> epoch: 0, batch index: 200, train loss: 0.232820
# ==>>> epoch: 0, batch index: 300, train loss: 0.181393
# ==>>> epoch: 0, batch index: 400, train loss: 0.125217
# ==>>> epoch: 0, batch index: 500, train loss: 0.093630
# ==>>> epoch: 0, batch index: 600, train loss: 0.101837
# 599
# ==>>> epoch: 0, batch index: 100, test loss: 0.064368, acc: 0.973
# 99