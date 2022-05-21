# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:40:31 2022

@author: 郭芳芳
"""
#%matplotlib inline
import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l
#根据带有噪音的线性模型构造一个人工数据集，y = xw+b+e
def synthetic_data (w,b,num_example):
    X = torch.normal(0,1,(num_example,len(w)))
    y = torch.matmul(X,w)+b
    y +=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))#让y变成一列

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
#验证一下得到的数据集
print("fearues:",features[0],'\nlabel:',labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
d2l.plt.show()

#定义一个data_iter 函数,该函数接收批量大小、特征矩阵和标签作为输入，生成大小为batch_size的小批量数据集
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,"\n",y)
    break

#定义初始化模型参数
w = torch.normal(0,0.01,size=(2,1),requires_grad = True)
b = torch.zeros(1,requires_grad = True)

#定义一个线性回归的模型
def linreg(X,w,b):
    return torch.matmul(X,w)+b
#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2
#定义优化算法
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()#把之前的梯度清零
            
#训练模型
lr = 0.03
num_epochs = 3 #把数据集扫3遍
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_1 = loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_1.mean()):f}')
        
#比较参数评估模型
print(f'w的估计误差:{true_w-w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b-b}')



#线性回归的简洁实现
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)

#构建一个pythorch数据迭代器
def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle = is_train)
batch_size=10
data_iter = load_array((features,labels),batch_size)
next(iter(data_iter))############

#使用框架的预定义好的层
from torch import nn
net = nn.Sequential(nn.Linear(2,1))
#初始化参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
#使用均方误差作为损失函数
loss = nn.MSELoss()
#实例化SGD
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)
#训练代码
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()#调用step进行模型的更新
    l = loss(net(features),labels)
    print(f"epoch{epoch+1},loss{l:f}")
#比较误差
w = net[0].weight.data
b = net[0].bias.data
print("w的估计误差", true_w -w.reshape(true_w.shape))
print("b的估计误差",true_b - b)




















