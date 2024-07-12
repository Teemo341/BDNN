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

parser = argparse.ArgumentParser(description='PyTorch Rank1BNN Training')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='imagenet', help='in domain dataset')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[30, 60, 90], nargs='+', help='decreasing strategy')
parser.add_argument('--seed', type=float, default=0)
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
model = models.Rank1BNN()
# net = torchvision.models.resnet18(pretrained=False,num_classes=200)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch in args.decreasing_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.droprate


if not os.path.isdir('./save_rank1bnn_imagenet'):
    os.makedirs('./save_rank1bnn_imagenet')
torch.save(model.state_dict(),'./save_rank1bnn_imagenet/final_model')