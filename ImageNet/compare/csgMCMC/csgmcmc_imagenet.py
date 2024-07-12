#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:34:10 2019

@author: lingkaikong
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import argparse
import models
import torchvision
import data_loader
import time
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='imagenet', help='in domain dataset')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[30, 60, 90], nargs='+', help='decreasing strategy')
parser.add_argument('--seed', type=float, default=0)
parser.add_argument('--temperature', type=float, default=1./50000,
                    help='temperature (default: 1/dataset_size)')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(device)

# Data
torch.manual_seed(args.seed)


if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)


print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)

# Model
print('==> Building model..')
net = models.Resnet()
# net = torchvision.models.resnet18(pretrained=False,num_classes=200)
net = net.to(device)

def noise_loss(lr,alpha):
    noise_loss = 0.0
    noise_std = (2/lr*alpha)**0.5
    for var in net.parameters():
        means = torch.zeros(var.size()).cuda()
        noise_loss += torch.sum(var * torch.normal(means, std = noise_std).cuda())
    return noise_loss

def adjust_learning_rate(optimizer, epoch, num_batch, batch_idx):
    T = args.epochs*num_batch
    M = 4
    lr_0 = args.lr
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    datasize = len(train_loader.dataset)
    num_batch = datasize/args.batch_size+1
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, epoch, num_batch, batch_idx)
        outputs = net(inputs)
        if (epoch%50)+1>47:
            loss_noise = noise_loss(lr,args.alpha)*(args.temperature/datasize)**.5
            loss = criterion(outputs, targets)+loss_noise
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx%100==0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, outputs = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx%100==0:
                print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss/len(test_loader), correct, total,
    100. * correct.item() / total))
           
mt = 0
for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    if (epoch%50)+1>47: # save 3 models per cycle
        print('save!')
        net.cpu()
        torch.save(net.state_dict(),'./save_csgmcmc_imagenet/imagenet_model_%i.pt'%(mt))
        mt +=1
        net.to(device)

