# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:58:54 2022

@author: 郭芳芳
"""
import torch
from IPython import display
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
import torchvision
from torch import nn

batch_size = 256
def get_dataloader_workers():
    return 4
def load_data_fashion_mnist(batch_size,resize = None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = "C:/Users/郭芳芳/data",train =True ,transform = trans,download = True)#将数据存储到上级目录的data中
    mnist_test = torchvision.datasets.FashionMNIST(root = "C:/Users/郭芳芳/data",train = False,transform = trans,download = True)
    return (data.DataLoader(mnist_train,batch_size,shuffle = True,num_workers = get_dataloader_workers()),
           data.DataLoader(mnist_test,batch_size,shuffle = True,num_workers = get_dataloader_workers()))

train_iter,test_iter = load_data_fashion_mnist(batch_size)

#softmax 回归的输出层是一个全连接层
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))
def init_weight(m):
    if type(m) == nn.Linear:#m表示当前的layer
        nn.init.normal_(m.weight,std = 0.01)
net.apply(init_weight)
#在交叉熵损失函数中传递没有归一化的预测值，并同时计算softmax及其对数
loss = nn.CrossEntropyLoss()
#使用学习率为0.1 的小批量随机梯度下降作为优化算法
trainer = torch.optim.SGD(net.parameters(),lr=0.1)
#调用之前定义的训练数据来训练模型
num_epochs = 10
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)



















