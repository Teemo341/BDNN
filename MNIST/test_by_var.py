###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################

from __future__ import print_function
import argparse
from itertools import Predicate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import calculate_log as callog
import models
import math
import os
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from numpy.linalg import inv
import numpy as np

# Training settings
parser = argparse.ArgumentParser(
    description='Test code - measure the detection peformance by variance')
parser.add_argument('--eva_iter', default=10, type=int,
                    help='number of passes when evaluation')
parser.add_argument('--network', type=str, choices=['resnet', 'resnet_bayesian',
                                                    'sdenet', 'mc_dropout', 'sdenet_multi', 'sdenet_bayesian'], default='resnet')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--dataset', required=True, help='in domain dataset')
parser.add_argument('--imageSize', type=int, default=28,
                    help='the height / width of the input image to network')
parser.add_argument('--out_dataset', required=True,
                    help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('--pre_trained_net', default='',
                    help="path to pre trained_net")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--test_batch_size', type=int, default=1000)


args = parser.parse_args()
print(args)

outf = 'test_by_var/'+args.network

if not os.path.isdir(outf):
    os.makedirs(outf)


device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if device == 'cuda':
    torch.cuda.manual_seed(args.seed)


print('Load model')
if args.network == 'resnet':
    model = models.Resnet()
    args.eva_iter = 1
elif args.network == 'resnet_bayesian':
    model = models.Resnet_bayesian()
elif args.network == 'sdenet':
    model = models.SDENet_mnist(layer_depth=6, num_classes=10, dim=64)
elif args.network == 'sdenet_multi':
    model = models.SDENet_multi_mnist(layer_depth=6, num_classes=10, dim=64)
elif args.network == 'mc_dropout':
    model = models.Resnet_dropout()


model.load_state_dict(torch.load(args.pre_trained_net))
model = model.to(device)
model_dict = model.state_dict()


print('load target data: ', args.dataset)
_, test_loader = data_loader.getDataSet(
    args.dataset, args.batch_size, args.test_batch_size, args.imageSize)

print('load non target data: ', args.out_dataset)
nt_train_loader, nt_test_loader = data_loader.getDataSet(
    args.out_dataset, args.batch_size, args.test_batch_size, args.imageSize)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def evaluate(regressor, data):
    batch_output = [regressor(data) for i in range(args.eva_iter)]
    batch_output = torch.stack(batch_output).float()
    means = batch_output.mean(axis=0)
    means = F.softmax(means)
    stds = batch_output.std(axis=0)
    return means, stds


def generate_target():
    model.eval()
    if args.network == 'mc-dropout':
        model.apply(apply_dropout)
    correct = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt' % outf, 'w')
    f3 = open('%s/confidence_Base_Succ.txt' % outf, 'w')
    f4 = open('%s/confidence_Base_Err.txt' % outf, 'w')

    with torch.no_grad():
        for data, targets in test_loader:
            total += data.size(0)
            data, targets = data.to(device), targets.to(device)
            batch_output = 0
            batch_std = 0
            batch_output,batch_std = evaluate(model,data)
            # compute the accuracy
            _, predicted = batch_output.max(1)
            correct += predicted.eq(targets).sum().item()
            correct_index = (predicted == targets)
            for i in range(data.size(0)):
                # confidence score: var_y
                # large var mean low confidence, invert var to keep pace with p(y|x)
                std = batch_std[i,predicted[i]].item()
                std = -std
                f1.write("{}\n".format(std))
                if correct_index[i] == 1:
                    f3.write("{}\n".format(std))
                elif correct_index[i] == 0:
                    f4.write("{}\n".format(std))
    f1.close()
    f3.close()
    f4.close()
    print('\n Final Accuracy: {}/{} ({:.2f}%)\n '.format(correct,
                                                         total, 100. * correct / total))


def generate_non_target():
    model.eval()
    if args.network == 'mc-dropout':
        model.apply(apply_dropout)
    total = 0
    f2 = open('%s/confidence_Base_Out.txt' % outf, 'w')
    with torch.no_grad():
        for data, targets in nt_test_loader:
            total += data.size(0)
            data, targets = data.to(device), targets.to(device)
            batch_output = 0
            batch_std = 0
            batch_output,batch_std = evaluate(model,data)
            _, predicted = batch_output.max(1)
            for i in range(data.size(0)):
                # confidence score: var_y
                # large var mean low confidence, invert var to keep pace with p(y|x)
                std = batch_std[i,predicted[i]].item()
                std = -std
                f2.write("{}\n".format(std))
    f2.close()


print('generate log from in-distribution data')
generate_target()
print('generate log  from out-of-distribution data')
generate_non_target()
print('calculate metrics for OOD')
callog.metric(outf, 'OOD')
print('calculate metrics for mis')
callog.metric(outf, 'mis')
