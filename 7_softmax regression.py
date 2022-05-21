# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:42:40 2022

@author: 郭芳芳
"""
import torch
from IPython import display
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

train_iter,test_iter = load_data_fashion_mnist(batch_size)#返回训练集和测试集的迭代器
#help(data.DataLoader)
#平铺每个图像，将其视为28*28=784的向量，因为有十个类别因此输出维度为10，
num_inputs = 784
num_outputs = 10

w = torch.normal(0,0.01,size = (num_inputs,num_outputs),requires_grad = True)
b = torch.zeros(num_outputs,requires_grad = True)

#实现softmax,其中x为矩阵,对每一行做softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim = True)
    return X_exp /partition

#验证一下softmax的正确性,按照概率原理，每一行总和为1
X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
X_prob,X_prob.sum(1)

#实现softmax回归的模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b)

##补充小知识：创建一个数据Y_hat,其中包含2个样本在3个类别的预测概率，使用Y作为Y_hat中概率的索引
y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y_hat[[0,1],y]

#实现交叉熵损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])#拿出真是标号的预测值

#将预测类别与真实类别进行比较
def accuracy(y_hat,y):
    """
    

    Parameters
    ----------
    y_hat : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    计算预测正确的数量

    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)#取每一行里面概率最大的下标
    cmp = y_hat.type(y.dtype) == y#把y_hat转换成y的数据类型
    return float(cmp.type(y.dtype).sum())
acc=accuracy(y_hat,y)/len(y)

#评估任意模型net的准确率

def evaluate_accuracy (net,data_iter):
    if isinstance(net,torch.nn.Module):#判断是否是一个torch nn实现的模型
        net.eval()#如果是的话，将模型设置为评估模式
    metric = Accumulator(2)#正确预测数、预测总数，它是一个累加器，不断地累加
    for X,y in data_iter:#对于迭代器来说，每一次拿到的一个batch 
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]
        
class Accumulator:
    '''
    在N个变量上累加
    '''
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
    
evaluate_accuracy(net, test_iter)

#softmax回归训练
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer ):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y),accuracy(y_hat,y),y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

#定义一个动画中绘制数据的实用程序类
class Animator:
    def __init__(self,xlabel = None,ylabel = None,legend = None,xlim = None,ylim = None,xscale = 'linear',
                 yscale = "linear",fmts = ('-','m--','g-.','r:'),nrows = 1,ncols = 1,figsize = (3.5,2.5)):
        if legend is None:
            legend=[]
        d2l.use_svg_display()
        self.fig,self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if nrows*ncols ==1:
            self.axes = [self.axes,]
        self.config_axes = lambda:d2l.set_axes(self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
        self.X,self.Y,self.fmts = None,None,fmts
    def add(self,x,y):
        if not hasattr(y,"__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x]*n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i ,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait = True)
        
#train function
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator= Animator(xlabel = "epoch",xlim = [1,num_epochs],ylim = [0.3,0.9],
                          legend=["train_loss","train_acc","test_acc"])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc = train_metrics
    assert train_loss < 0.5,train_loss
    assert train_acc <=1 and train_acc >0.7,train_acc
    assert test_acc <=1 and test_acc >0.7,test_acc
    
lr = 0.1
def updater(batch_size):
    return d2l.sgd([w,b],lr,batch_size)
num_epochs = 10
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)
        
    
#用训练好的模型来预测

def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis =1))
    titles = [true+'\n'+pred for true,pred in zip(trues,preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)), 1,n,titles = titles[0:n])
predict_ch3(net,test_iter)



















