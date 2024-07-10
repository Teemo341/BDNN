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

from models.swag.posteriors import SWAG
from models.swag import utils




parser = argparse.ArgumentParser(description='PyTorch SWAG Training')
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

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu') # type: ignore
print(device)

# Data
torch.manual_seed(args.seed)


if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed) # type: ignore


print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)

# Model
print('==> Building model..')
net = models.Resnet()
net = net.to(device)

print("SWAG training")
swag_model = SWAG(
    net,
    no_cov_mat=False,
    max_num_models=20,
    num_classes=200,
)
swag_model.to(args.device)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0
swa_start = args.epochs//2


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    time_ = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    train_res = utils.train_epoch(train_loader, net, criterion, optimizer)

    if (epoch + 1) > swa_start:
        # sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
        sgd_res = utils.predict(test_loader, net)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        print("updating sgd_ens")

        global sgd_ens_preds
        global n_ensembled
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
            ) + sgd_preds / (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(net)
    
        
    print('Train epoch:{} \tLoss: {:.6f} | Acc: {:.6f} ({}/{}) | Time: {:.2f}m'
        .format(epoch, train_loss/(len(train_loader)), 100.*correct/total, correct, total, (time.time()-time_)/60))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_res = utils.eval(test_loader, net, criterion)

    swag_model.sample(0.0)
    utils.bn_update(train_loader, swag_model)
    swag_res = utils.eval(test_loader, swag_model, criterion)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


        print('Test epoch: {}  Acc: {} ({}/{})'
        .format(epoch, 100.*correct/total, correct, total))
           

for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch in args.decreasing_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.droprate


if not os.path.isdir('./save_resnet_imagenet'):
    os.makedirs('./save_resnet_imagenet')
utils.save_checkpoint('./save_resnet_imagenet', epoch, state_dict = net.state_dict(),optimizer = optimizer.state_dict())
utils.save_checkpoint('./save_resnet_imagenet', epoch, name='swag', state_dict = swag_model.state_dict())


