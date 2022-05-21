# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:45:05 2022

@author: 郭芳芳
"""
#自动求导
import torch
x = torch.arange(4.0)
#设置存储梯度
x.requires_grad_(True)
x.grad

y = 2*torch.dot(x,x)
#调用反向传播函数计算y关于x的每个分量的梯度
y
y.backward()
x.grad
x.grad == 4*x

#将之前的梯度清零
x.grad.zero_()
y1 = x.sum()
y1.backward()
x.grad

