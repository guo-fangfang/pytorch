# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:53:10 2022

@author: 郭芳芳
"""
import torch
x = torch.arange(12)
print(x)
#x的属性
print(x.shape)
print(x.numel())#元素个数
y = x.reshape(3,4)
print(y)

x1  = torch.zeros((2,3,4))
x2 = torch.ones((2,3,4))
print(x1,"\n")
print(x2)

#创建torch张量
x3 = torch.tensor([[1,2,3],[4,5,6]])
print(x3)
x4 = [[7,8,9],[10,11,12]]
x4 = torch.tensor(x4)
print(x4)
print(type(x4))

#简单的运算
x = torch.tensor([1,2,3,4],dtype = torch.float32)
y = torch.tensor([5,6,7,8],dtype = torch.float32)
z1 = x+y
z2 = x-y
z3 =x*y
z4 = x/y
z5 = y**x
print("z1",z1)
print("z2",z2)
print("z3",z3)
print("z4",z4)
print("z5",z5)

#将多个张量连接在一起
x = torch.arange(12,dtype = torch.float32).reshape((3,4))
y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]],dtype = torch.float32)
z1 = torch.cat((x,y),dim=0)
z2 = torch.cat((x,y),dim=1)
print("z1:","\n",z1)
print("z2:","\n",z2)

x.sum()
#元素的访问、
s1=x[-1]
s2=x[-1,:]
print(s1)
print("\n",s2)
s1==s2











