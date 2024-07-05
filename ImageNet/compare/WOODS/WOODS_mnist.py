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
import data_loader
import time
import numpy as np
import torch.nn.functional as F

print(torch.cuda.is_available())

parser = argparse.ArgumentParser(description='PyTorch ResNet WOODS Training')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', default='svhn', help='cifar10 | svhn')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[10,20,30], nargs='+', help='decreasing strategy')
parser.add_argument('--seed', type=float, default=0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(args.seed)

torch.manual_seed(args.seed)

if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)

# Model
print('==> Building model..')
net = models.Resnet_aux()
net = net.to(device)

fake_label = 1/10


optimizer = optim.SGD(
        list(net.parameters()),
        args.lr, momentum=0.9,
        weight_decay=0.0005, nesterov=True)

in_constraint_weight=1.0
ce_constraint_weight = 1.0
lam = torch.tensor(0).float()
lam = lam.cuda()
lam2 = torch.tensor(0).float()
lam2 = lam.cuda()



# Training
def train(epoch):
    global in_constraint_weight
    global ce_constraint_weight
    global lam
    global lam2
    start = time.time()
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss_in_drift = 0
    train_loss_in_diffu = 0
    train_loss_out_diffu = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #make data
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_in = inputs[:int(args.batch_size/2),]
        aux_in = inputs[int(args.batch_size/2):,]
        targets_in = targets[:int(args.batch_size/2),]
        aux_out = 16*torch.randn(len(aux_in),1, args.imageSize, args.imageSize, device = device)+aux_in
        inputs_aux = mix_batches(aux_in,aux_out)
        inputs = torch.cat((inputs_in, inputs_aux), 0)
        targets = targets_in

        outputs = net(inputs)
        outputs = outputs.to(device)
        outputs_classification = outputs[:len(inputs_in), :10]

        loss_ce = F.cross_entropy(outputs_classification, targets)

        out_x_ood_task = outputs[len(inputs_in):, 10]
        out_loss = torch.mean(F.relu(1 - out_x_ood_task))
        out_loss_weighted = 1* out_loss

        in_x_ood_task = outputs[:len(inputs_in), 10]
        f_term = torch.mean(F.relu(1 + in_x_ood_task)) - 0.05
        if in_constraint_weight * f_term + lam >= 0:
            in_loss = f_term * lam + in_constraint_weight / 2 * torch.pow(f_term, 2)
        else:
            in_loss = - torch.pow(lam, 2) * 0.5 / in_constraint_weight

        #? seems better
        # loss_ce_constraint = loss_ce - 2 * full_train_loss
        # if ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
        #     loss_ce = loss_ce_constraint * lam2 + ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
        # else:
        #     loss_ce = - torch.pow(lam2, 2) * 0.5 / ce_constraint_weight

        # add the losses together
        loss = loss_ce + out_loss_weighted + in_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_in_drift += loss_ce.item()
        train_loss_in_diffu += in_loss.item()
        train_loss_out_diffu += out_loss_weighted.item()
        _, predicted = outputs_classification.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # updates for alm methods
    print("making updates for SSND alm methods...")

    # compute terms for constraints
    in_term, ce_loss = compute_constraint_terms()

    # update lam for in-distribution term
    print("updating lam...")
    in_term_constraint = in_term - 0.05
    print("in_distribution constraint value {}".format(in_term_constraint))
    # update lambda
    if in_term_constraint * in_constraint_weight + lam >= 0:
        lam += 1 * in_term_constraint
    else:
        lam += -1 * lam / in_constraint_weight

    # update lam2
    print("updating lam2...")
    ce_constraint = ce_loss - 2 * full_train_loss
    print("cross entropy constraint {}".format(ce_constraint))
    # update lambda2
    if ce_constraint * ce_constraint_weight + lam2 >= 0:
        lam2 += 1 * ce_constraint
    else:
        lam2 += -1 * lam2 / ce_constraint_weight

    # update weight for alm_full_2
    if in_term_constraint > 0:
        print('increasing in_constraint_weight weight....\n')
        in_constraint_weight *= 1.5

    if ce_constraint > 0:
        print('increasing ce_constraint_weight weight....\n')
        ce_constraint_weight *= 1.5
        
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
            outputs = outputs[:len(inputs), :10]
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test epoch: {}| Acc: {} ({}/{}) '
        .format(epoch, 100.*correct/total, correct, total))
           
def mix_batches(aux_in_set, aux_out_set):
    '''
    Args:
        aux_in_set: minibatch from in_distribution
        aux_out_set: minibatch from out distribution

    Returns:
        mixture of minibatches with mixture proportion pi of aux_out_set
    '''

    # create a mask to decide which sample is in the batch
    pi = 0.1
    mask = rng.choice(a=[False, True], size=(len(aux_in_set),), p=[1 - pi, pi])

    aux_out_set_subsampled = aux_out_set[mask]
    aux_in_set_subsampled = aux_in_set[np.invert(mask)]

    # note: ordering of aux_out_set_subsampled, aux_in_set_subsampled does not matter because you always take the sum
    aux_set = torch.cat((aux_out_set_subsampled, aux_in_set_subsampled), 0)

    return aux_set

def to_np(x): return x.data.cpu().numpy()

def evaluate_classification_loss_training():
    '''
    evaluate classification loss on training dataset
    '''
    net.eval()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(device)
        target=target.to(device)
        data = data[:int(args.batch_size/2),]
        target = target[:int(args.batch_size/2),]

        # forward
        x = net(data)

        # in-distribution classification accuracy
        x_classification = x[:,:10]
        loss_ce = F.cross_entropy(x_classification, target, reduction='none')

        losses.extend(list(to_np(loss_ce)))

    avg_loss = np.mean(np.array(losses))
    print("average loss fr classification {}".format(avg_loss))

    return avg_loss

full_train_loss = evaluate_classification_loss_training()

def compute_constraint_terms():
    '''

    Compute the in-distribution term and the cross-entropy loss over the whole training set
    '''

    net.eval()

    # create list for the in-distribution term and the ce_loss
    in_terms = []
    ce_losses = []
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        num_batches += 1
        data=data.to(device)
        target=target.to(device)
        data = data[:int(args.batch_size/2),]
        target = target[:int(args.batch_size/2),]

        # forward
        net(data)
        z = net(data)

        # compute in-distribution term
        in_x_ood_task = z[:, 10]
        in_terms.extend(list(to_np(F.relu(1 + in_x_ood_task))))

        # compute cross entropy term
        z_classification = z[:, :10]
        loss_ce = F.cross_entropy(z_classification, target, reduction='none')
        ce_losses.extend(list(to_np(loss_ce)))

    return np.mean(np.array(in_terms)), np.mean(np.array(ce_losses))


for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch in args.decreasing_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.droprate

if not os.path.isdir('./save_resnet_WOODS_mnist'):
    os.makedirs('./save_resnet_WOODS_mnist')
torch.save(net.state_dict(),'./save_resnet_WOODS_mnist/final_model')

