# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:09:32 2022

@author: 郭芳芳
"""
#标量
import torch
x = torch.tensor([3],dtype=torch.float32)
y = torch.tentor([2],dtype=torch.float32)
#向量
x = torch.arange(12)
x[2]
len(x)
x.shape
#矩阵
A = torch.arange(20).reshape(5,4)
B = A.T
print(A)
print(B)
C=A.clone()
print(C)
print(A*C)
A.sum()
A.sum(axis=0)
A.sum(axis=1)
#保持维度一致
sum_A = A.sum(axis = 1,keepdims = True)
#累加求和
cumsum_A = A.cumsum(axis=1)
print(cumsum_A)
A
#向量点积
x = torch.tensor([1,2,3,4])
y= torch.tensor([5,6,7,8])
z = torch.dot(x,y)
print(z)
#矩阵和向量的乘积
z1 = torch.mv(A,x)
print(z1)
#矩阵和矩阵的乘积
z2 = torch.mm(A,B)
print(z2)
#L2范数
u = torch.tensor([3.0,-4.0])
u_l2 = torch.norm(u)
print(u_l2)
#L1范数
u_l1 = torch.abs(u).sum()
#矩阵的F范数:矩阵元素平方和的平方根
F = torch.norm(torch.ones((4,9)))
#按照特定轴求和，按照哪个轴就把哪个轴元素去掉，若保留dims那么那一维就变成了1
x = torch.arange(40).reshape(2,5,4)
y1 = torch.sum(x,axis =0)
y2 = torch.sum(x,axis =1)
y3 = torch.sum(x,axis = 2)
y4 = torch.sum(x,axis =[0,1])

print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y4.shape)

y5 = torch.sum(x,axis=0,keepdims = True)
print(y5.shape)










