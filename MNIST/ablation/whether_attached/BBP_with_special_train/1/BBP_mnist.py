#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import argparse
import models
import data_loader
import time

print(torch.cuda.is_available())

parser = argparse.ArgumentParser(description='PyTorch BBP Training')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='mnist', help='cifar10 | svhn')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[10, 20, 30], nargs='+', help='decreasing strategy')
parser.add_argument('--seed', type=float, default=0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(args.seed)

if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)

# Model
print('==> Building model..')
net = models.BBP()
net = net.to(device)


fake_label = 1/10


# optimizer1 = optim.Adam([{'params':[ param for name, param in net.named_parameters() if 'bayesian' not in name]}], lr=0.001)
# optimizer2 = optim.Adam([{'params':[ param for name, param in net.named_parameters() if 'bayesian' in name]}], lr=0.001)
optimizer2 = optim.Adam(net.parameters(), lr=0.001)
criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.BCEWithLogitsLoss()


# Training
def train(epoch):
    start = time.time()
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss_in_drift = 0
    train_loss_in_diffu = 0
    train_loss_out_diffu = 0
    correct = 0
    total = 0
    ##training with in-domain data
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # optimizer1.zero_grad()
        optimizer2.zero_grad()
        outputs = net(inputs)
        outputs = outputs.to(device)
        # loss1 = criterion1(outputs,targets)
        # loss1.backward()
        # optimizer1.step()
        loss2 = net.sample_elbo(inputs=inputs,
                                       labels=targets,
                                       criterion=criterion1,
                                       sample_nbr=3,
                                       complexity_cost_weight=1 / 50000)
        loss2.backward()
        optimizer2.step()
        # train_loss_in_drift += loss1.item()
        train_loss_in_diffu += loss2.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    ##training with out-of-domain data
        optimizer2.zero_grad()
        label = torch.full((args.batch_size,10), fake_label, device=device)
        label = label.to(device)
        inputs_out = 16*torch.randn(args.batch_size,1, args.imageSize, args.imageSize, device = device)+inputs
        inputs_out = inputs_out.to(device)
        loss_out = net.sample_elbo(inputs=inputs_out,
                                       labels=label,
                                       criterion=criterion2,
                                       sample_nbr=3,
                                       complexity_cost_weight=1 / 50000)
        train_loss_out_diffu += (loss_out.item())
        loss_out = -1*loss_out
        loss_out.backward()
        optimizer2.step()
        
    print('Train epoch:{} \tLoss_in: {:.6f} | Loss_in_diffu: {:.6f} | Loss_out_diffu: {:.6f} | Acc: {:.6f} ({}/{})'
        .format(epoch, train_loss_in_drift/(len(train_loader)),train_loss_in_diffu/(len(train_loader)), train_loss_out_diffu/(len(train_loader)),100.*correct/total, correct, total))
    end = time.time()
    print((start,end,end-start))

def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).to(device)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test epoch: {}| Acc: {} ({}/{}) '
        .format(epoch, 100.*correct/total, correct, total))
           


for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch in args.decreasing_lr:
        # for param_group in optimizer1.param_groups:
            # param_group['lr'] *= args.droprate
        for param_group in optimizer2.param_groups:
            param_group['lr'] *= args.droprate


if not os.path.isdir('./save_BBP_mnist'):
    os.makedirs('./save_BBP_mnist')
torch.save(net.state_dict(),'./save_BBP_mnist/final_model')

