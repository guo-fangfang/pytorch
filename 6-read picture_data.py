# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:54:35 2022

@author: 郭芳芳
"""
import matplotlib.pyplot as plt
import torch 
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
#通过totrnsor实例将图片数据从pil类型转换成32位浮点数格式
#并且除以255使得所有像素的数值均位于0-1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root = "C:/Users/郭芳芳/data",train =True ,transform = trans,download = True)#将数据存储到上级目录的data中
mnist_test = torchvision.datasets.FashionMNIST(root = "C:/Users/郭芳芳/data",train = False,transform = trans,download = True)
len(mnist_train)
len(mnist_test)
type(mnist_train)
mnist_train[0][0].shape#第一个样本的数据#灰色图片，只有一个channel，长和宽都是28
#定义可视化函数
def get_fashion_mnist_labels(labels):
    """
    Parameters
    ----------
    lables : string
    Returns  
    -------
    返回数据集的文本标签

    """
    text_labels = ["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows,num_cols,titles = None,scale = 1.5):
    """
    Parameters
    ----------
    imgs : TYPE
        DESCRIPTION.
    num_rows : TYPE
        DESCRIPTION.
    num_cols : TYPE
        DESCRIPTION.
    titles : TYPE, optional
        DESCRIPTION. The default is None.
    scale : TYPE, optional
        DESCRIPTION. The default is 1.5.

    Returns
    -------
    plot a list of images

    """
    
    figsize = (num_cols*scale,num_rows*scale)
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize = figsize)
    axes = axes.flatten()########将ndarray变成一行数组
    for i ,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X,y = next(iter(data.DataLoader(mnist_train,batch_size = 18)))
print(X)
X.shape
y
show_images(X.reshape(18,28,28),2,9,titles = get_fashion_mnist_labels(y))
    
    
    
#读取小批量数据
batch_size = 256
def get_dataloader_workers():
    return 4

train_iter = data.DataLoader(mnist_train,batch_size,shuffle = True,num_workers = get_dataloader_workers())

timer = d2l.Timer()
for x,y in train_iter:
    continue
f"{timer.stop():.2f}sec"
    
 #定义一个data load function

def load_data_fashion_mnist(batch_size,resize = None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = "C:/Users/郭芳芳/data",train =True ,transform = trans,download = True)#将数据存储到上级目录的data中
    mnist_test = torchvision.datasets.FashionMNIST(root = "C:/Users/郭芳芳/data",train = False,transform = trans,download = True)

   return (data.DataLoader(mnist_train.batch_size,shuffle = True,num_workers = get_dataloader_workers()),
           data.DataLoader(mnist_test.batch_size,shuffle = True,num_workers = get_dataloader_workers()))
train_iter ,test_iter = load_data_fashion_mnist(32,resize = 64)
for x,y train_iter:
    print(x.shape,x.dtype,y,shape,y.dtype)
    
    
    
    
    
    
    
    
    
