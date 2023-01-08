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

parser = argparse.ArgumentParser(description='PyTorch VBOE Training')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='svhn', help='cifar10 | svhn')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
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


optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.BCEWithLogitsLoss()


# Training
def train(epoch):
    start = time.time()
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss_in = 0
    train_loss_out = 0
    correct = 0
    total = 0
    ##training with in-domain data
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        ##training with in-domain data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        outputs = outputs.to(device)
        loss1 = net.sample_elbo(inputs=inputs,
                                       labels=targets,
                                       criterion=criterion,
                                       sample_nbr=3,
                                       complexity_cost_weight=1 / 50000)
        train_loss_in += loss1.item()
        
        #training with ood data
        label = torch.full((args.batch_size,10), fake_label, device=device)
        label = label.to(device)
        inputs_out = 16*torch.randn(args.batch_size,1, args.imageSize, args.imageSize, device = device)+inputs
        inputs_out = inputs_out.to(device)
        outputs_out = outputs = net(inputs_out)
        loss2 = criterion2(outputs_out,label)
        train_loss_out+= loss2.item()

        loss = loss1+loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    print('Train epoch:{} \tLoss_in: {:.6f} | Loss_out: {:.6f} | Acc: {:.6f} ({}/{})'
        .format(epoch, train_loss_in/(len(train_loader)), train_loss_out/(len(train_loader)),100.*correct/total, correct, total))
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
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.droprate


if not os.path.isdir('./save_VBOE_mnist'):
    os.makedirs('./save_VBOE_mnist')
torch.save(net.state_dict(),'./save_VBOE_mnist/final_model')

