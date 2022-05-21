# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:19:05 2022

@author: 郭芳芳
"""
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

#首先生成一个多项式数据集
max_degree = 20#表示多项式的阶最多有20个，在深度学习中表示要训练的模型参数有20个
n_train,n_test = 100,100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])#通过这两步操作实现了前4个参数不为0，后面的参数全部为0

features = np.random.normal(size = (n_train+n_test,1))#生成一个行为200 列为1 的正态分布的二维数组
np.ranndom.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
poly_features.shape#得到了200个20列的矩阵X
#print(poly_features)
for i in range(max_degree):
    poly_features[:,i] = poly_features[:,i]/math.gamma(i+1)
labels = np.dot(poly_features,true_w)
labels = labels+np.random.normal(scale = 0.1,size = labels.shape)

true_w,features,poly_features,labels = [torch.tensor(x,dtype = torch.float32)
                                        for x in [true_w,features,poly_features,labels]]

#将numpy 转化为tensor 
#查看一下数据
features[:2],poly_features[:2,:],labels[:2]

#实现一个函数来评估模型在给定数据集上的损失

def evaluate_loss (net,data_iter,loss):
    metric = d2l.Accumulator(2)
    for X,y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]

#定义训练函数
def train(train_features,test_features,train_labels,test_labels,num_epochs = 300):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]#shape 是20
    net = nn.Sequential(nn.Linear(input_shape,1,bias = False))#输入的维度（列）是20，输出是1
    batch_size = min(10,train_labels.shape[0])
    train_iter = d2l.load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features,test_labels.reshape(-1,1)),batch_size,is_train = False)
    trainer = torch.optim.SGD(net.parameters(),lr = 0.01)
    animator = d2l.Animator(xlabel = 'epoch',ylabel = "loss",yscale = "log",xlim = [1,num_epochs],ylim = [1e-3,1e2],legend = ["train","test"])
    
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net,train_iter,loss,trainer)
        if epoch==0 or (epoch+1)%20 ==0:
            animator.add(epoch+1,(evaluate_loss(net,train_iter,loss),evaluate_loss(net,test_iter,loss)))
    print("weight",net[0].weight.data.numpy())
    
#训练模型-没有出现过拟合和欠拟合
train(poly_features[:n_train,:4],poly_features[n_train:,:4],
      labels[:n_train],labels[n_train:])
#模型的输入是包含了非零阶权重的X

#训练一个欠拟合的模型,即输入的数据太简单了，没有抓取到足够的特征

train(poly_features[:n_train,:2],poly_features[n_train:,:2],
      labels[:n_train],labels[n_train:])

    
#过拟合即输入了太多无用的包含了很多噪音的数据
train(poly_features[:n_train,:],poly_features[n_train:,:],
      labels[:n_train],labels[n_train:],num_epochs=1500)




















