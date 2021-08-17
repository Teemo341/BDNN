import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
#%%
data_tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)
#%%
train_dataset = datasets.MNIST(root='C:/Users/83883/Desktop/论文/毕业论文/SDE-Net-master/data/mnist',train=True,transform=data_tf,download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=5,
                                           shuffle=True)
#%%
# 实现单张图片可视化
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images[0])

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print(labels)
plt.imshow(img)
plt.show()
