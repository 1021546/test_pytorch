# Author: Robert Guthrie
# 作者：Robert Guthrie
 
import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块
 
torch.manual_seed(1)

# lstm单元输入和输出维度都是3
lstm = nn.LSTM(3, 3) 
# 生成一个长度为5，每一个元素为1*3的序列作为输入，这里的数字3对应于上句中第一个3
inputs = [autograd.Variable(torch.randn((1, 3)))
for _ in range(5)]
 
# 设置隐藏层维度，初始化隐藏层的数据
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
autograd.Variable(torch.randn((1, 1, 3))))
 
for i in inputs:
# Step through the sequence one element at a time.
# after each step, hidden contains the hidden state.
	out, hidden = lstm(i.view(1, 1, -1), hidden)
	print(out)
	print(hidden)
