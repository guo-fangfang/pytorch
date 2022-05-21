# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:02:06 2022

@author: 郭芳芳
"""
#创建一个数据集并保存
import os
#print(os.path)
os.makedirs(os.path.join('..',"data"),exist_ok=True)
data_file = os.path.join("..","data","house_tiny.csv")
with open(data_file,"w") as f:
    f.write("numrooms,alley,price\n")
    f.write("NA,pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,14000\n")
#读取文件
import pandas as pd
data1= pd.read_csv(data_file)
print(data1,"\n")

data2 = pd.read_csv(os.path.join("..","data","house_tiny.csv"))
print(data2)
 
#处理缺失值
#插值法
inputs,outputs = data1.iloc[:,:2],data1.iloc[:,2]
print("inputs:",inputs)
print("outputs")
print(outputs)
inputs_1 = inputs.fillna(inputs.mean())
inputs_2 = inputs.fillna(inputs["numrooms"].mean())
print("inputs_1")
print(inputs_1)
print("\ninputs_2")
print(inputs_2)
#对于非数值的缺失值处理
#使用类似于onehot的处理,对每个样本
inputs_1 = pd.get_dummies(inputs_1,dummy_na = True)
print(inputs_1)
#没有缺失值之后将其转化为torch张量 
import torch 
x,y = torch.tensor(inputs_1.values),torch.tensor(outputs.values)#如果原本是DataFrame 则写成values

print("x\n",x,"\n")
print("y\n",y,"\n")

import numpy as np
X,Y = np.array(inputs_1),np.array(outputs)
type(X)
X,Y = torch.tensor(X),torch.tensor(Y)
print(X)
print(Y)

