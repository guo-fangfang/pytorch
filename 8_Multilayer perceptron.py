# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:56:51 2022

@author: 郭芳芳
"""
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
import torchvision

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

#实现一个具有单影藏层的多层感知机，它包含了256个隐藏单元

num_inputs,num_outputs,num_hiddens = 784,10,256
W1 = nn.Parameter(
    torch.randn(num_inputs,num_hiddens,requires_grad = True)*0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad = True))
W2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad = True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad = True))
params = [W1,b1,W2,b2]
#difine relu function
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

#model
def net(X):
    X = X.reshape((-1,num_inputs))#将数据转化成一个每一行都是一个向量的二维数组
    H = relu(X@W1+b1)
    return (H@W2+b2)
loss = nn.CrossEntropyLoss()
#train model
num_epochs,lr = 10,0.1
updater = torch.optim.SGD(params,lr = lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)


#简洁实现##################################################


net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLu(),nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std = 0.01)
net.apply(init_weights)

batch_size,lr,num_epochs = 256,0.1,10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(),lr = lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)














