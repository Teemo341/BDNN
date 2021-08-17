###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################
#%%
from __future__ import print_function
import visualization
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import torch.utils.data as Data
import calculate_log as callog
import models
import math
import os
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
#%%
# Training settings
# parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
# parser.add_argument('--eva_iter', default=10, type=int, help='number of passes when evaluation')
# parser.add_argument('--network', type=str, choices=['resnet', 'sdenet','mc_dropout'], default='resnet')
# parser.add_argument('--batch-size', type=int, default=256, help='batch size')
# parser.add_argument('--seed', type=int, default=0,help='random seed')
# parser.add_argument('--dataset',default='mnist', required=True, help='in domain dataset')
# parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
# parser.add_argument('--out_dataset', default= 'svhn',required=True, help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
# parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
# parser.add_argument('--pre_trained_net', default='', help="path to pre trained_net")
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--test_batch_size', type=int, default=1000)

#%%

outf = 'vis/'+'resnet'

if not os.path.isdir(outf):
    os.makedirs(outf)
seed = 314

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("Random Seed: ", seed)
torch.manual_seed(seed)

if device == 'cuda':
    torch.cuda.manual_seed(seed)

#%%
print('Load model')
network = 'resnet'
eva_iter = 10
if network == 'resnet':
    model = models.Resnet()
    eva_iter = 1
elif network == 'sdenet':
    model = models.SDENet_mnist(layer_depth=6, num_classes=10, dim=64)
elif network == 'mc_dropout':
    model = models.Resnet_dropout()


pre_trained_net = "save_resnet_mnist/final_model"
model.load_state_dict(torch.load(pre_trained_net))
model = model.to(device)
model_dict = model.state_dict()

#%%
print("load image")
dataset = "mnist"
batch_size = 256
test_batch_size = 1
imageSize = 28
print('load target data: ',dataset)
train_loader, test_loader = data_loader.getDataSet(dataset, batch_size, test_batch_size, imageSize)
images, labels = next(iter(test_loader))
#%%
# def apply_dropout(m):
#     if type(m) == nn.Dropout:
#         m.train()
#%%
print("load image")
# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#     transforms.Normalize([0.5],[0.5])])
#
# test_data = datasets.MNIST(
#     root='C:/Users/83883/Desktop/论文/毕业论文/SDE-Net-master/data/mnist',
#     train=False,
#     transform=data_tf
# )
# test_loader = torch.utils.data.DataLoader(dataset=test_data,
#                                            batch_size=1,
#                                            shuffle=True)
# images, labels = next(iter(test_loader))
# img = vutils.make_grid(images)
#
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
# img = torch.from_numpy(img)
# # print(labels)
# # plt.imshow(img)
# # plt.show()
#%%
with torch.no_grad():
    visualization.get_feature(images,network,pre_trained_net)
