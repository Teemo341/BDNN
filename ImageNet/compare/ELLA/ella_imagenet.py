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
import copy
from functools import partial
import numpy as np
import torch.nn.functional as F

from models.ELLA.data import subsample
from models.ELLA.utils import count_parameters, _ECELoss, build_dual_params_list, jac, Psi_raw, psd_safe_cholesky, ConvNet, check_approx_error, measure_speed

parser = argparse.ArgumentParser(description='LLA for DNNs')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--not-use-gpu', action='store_true', default=False)

parser.add_argument('-b', '--batch-size', default=100, type=int,
					metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--test-batch-size', default=128, type=int)

parser.add_argument('--arch', type=str, default='cifar10_resnet20')
parser.add_argument('--dataset', type=str, default='cifar10',
					choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data-root', type=str, default=None)
parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
					help='path to pretrained MAP checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='./logs/', type=str)
parser.add_argument('--job-id', default='default', type=str)
parser.add_argument('--resume-dual-params', default=None, type=str)
parser.add_argument('--resume-cov-inv', default=None, type=str)

parser.add_argument('--K', default=20, type=int)
parser.add_argument('--M', default=100, type=int, help='the number of samples')
parser.add_argument('--I', default=1, type=int, help='the number of samples')
parser.add_argument('--balanced', action='store_true', default=False)
parser.add_argument('--not-random', action='store_true', default=False)

# parser.add_argument('--early-stop', default=None, type=int)
parser.add_argument('--search-freq', default=None, type=int)
parser.add_argument('--sigma2', default=0.1, type=float)

parser.add_argument('--num-samples-eval', default=512, type=int,metavar='N')
parser.add_argument('--ntk-std-scale', default=1, type=float)

parser.add_argument('--check', action='store_true', default=False)
parser.add_argument('--measure-speed', action='store_true', default=False)
parser.add_argument('--track-test-results', action='store_true', default=False)


def main():
	args = parser.parse_args()
	args.save_dir = './save_ella_imagenet'
	args.num_classes = 200
	args.random = not args.not_random

	if args.M < args.num_classes:
		args.balanced = False

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	if not args.not_use_gpu and torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cudnn.benchmark = True
	else:
		device = torch.device('cpu')

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	train_loader_noaug, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)
	val_loader = test_loader

	model = models.Resnet().to(device)

	params = {name: p for name, p in model.named_parameters()}
	print("Number of parameters", count_parameters(model))
	if args.pretrained is not None:
		print("Load MAP model from", args.pretrained)
		model.load_state_dict(torch.load(args.pretrained))

	model.eval()
	print("---------MAP model ---------")
	test(test_loader, model, device, args)

	model_bk = copy.deepcopy(model)

	###### ella ######
	## check the approximation error
	if args.check:
		check_approx_error(args, model, params, train_loader_noaug, val_loader, device)

	if args.resume_dual_params is not None:
		if args.resume_dual_params == 'auto':
			args.resume_dual_params = os.path.join(args.save_dir, 'dual_params.tar.gz')
		ckpt = torch.load(args.resume_dual_params)
		dual_params_list = [{k:v.to(device) for k,v in dual_params.items()} for dual_params in ckpt['0']]
	else:
		x_subsample, y_subsample = subsample(train_loader_noaug, args.num_classes,
											 args.M, args.balanced,
											 device, verbose=False)
		dual_params_list = build_dual_params_list(model, params, x_subsample, y_subsample, args=args, num_batches=args.I)

		torch.save({
			'0': [{k:v.data.cpu() for k,v in dual_params.items()} for dual_params in dual_params_list],
		}, os.path.join(args.save_dir if not 'vit' in args.arch else '../ella_logs', 'dual_params.tar.gz'))

	if args.measure_speed:
		x, y = subsample(train_loader_noaug, args.num_classes, args.batch_size, False, device, verbose=False)
		measure_speed(model, params, dual_params_list, model_bk, x, y)

	Psi = partial(Psi_raw, model, params, dual_params_list)

	if args.resume_cov_inv is not None:
		if args.resume_cov_inv == 'auto':
			args.resume_cov_inv = os.path.join(args.save_dir, 'cov_inv.tar.gz')
		ckpt = torch.load(args.resume_cov_inv)
		best_cov_inv = ckpt['0'].to(device)
	else:
		## pass the training set
		best_value = 1e8; best_cov_inv = None; best_test_results = None
		test_results_list = []
		with torch.no_grad():
			cov = torch.zeros(args.K, args.K).cuda(non_blocking=True)
			for i, (x, y) in enumerate(train_loader_noaug):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
				Psi_x, logits = Psi(x, return_output=True)
				prob = logits.softmax(-1)
				Delta_x = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
				cov += torch.einsum('bok,boj,bjl->kl', Psi_x, Delta_x, Psi_x)

				if args.search_freq and (i + 1) * args.batch_size % args.search_freq == 0:
					cov_clone = cov.data.clone()
					cov_clone.diagonal().add_(1 / args.sigma2 * (i + 1) * args.batch_size / len(train_loader_noaug.dataset))
					cov_inv = cov_clone.inverse()
					val_loss, _, _ = ella_test(val_loader, model, device, args, Psi, cov_inv, verbose=True)
					if args.track_test_results:
						test_loss, test_acc, test_ece = ella_test(test_loader, model, device, args, Psi, cov_inv, verbose=False)
						test_results_list.append(np.array([(i + 1) * args.batch_size, test_loss, test_acc, test_ece]))
					if val_loss < best_value:
						best_value = val_loss
						best_cov_inv = cov_inv
						test_loss, test_acc, test_ece = ella_test(test_loader, model, device, args, Psi, best_cov_inv, verbose=False)
						best_test_results = "Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(test_loss, test_acc, test_ece)

						torch.save({
							'0': best_cov_inv.data.cpu(),
						}, os.path.join(args.save_dir, 'cov_inv.tar.gz'))

					print("Current training data {}, loss: {:.4f}, best loss: {:.4f}"
						  "\n    {}".format((i + 1) * args.batch_size, val_loss, best_value, best_test_results))

		if args.track_test_results:
			test_results_list = np.stack(test_results_list)
			np.save(args.save_dir + '/test_results.npy', test_results_list)


def ella_test(test_loader, model, device, args, Psi, cov_inv, verbose=True, return_more=False):
	t0 = time.time()
	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x, y in test_loader:
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			Psi_x, y_pred = Psi(x, return_output=True)
			F_var = Psi_x @ cov_inv.unsqueeze(0) @ Psi_x.permute(0, 2, 1)
			# if 'vit' in args.arch:
			# 	F_var_L = torch.stack([psd_safe_cholesky(item) for item in F_var])
			# else:
			F_var_L = psd_safe_cholesky(F_var)
			F_samples = (F_var_L @ torch.randn(F_var.shape[0], F_var.shape[1], args.num_samples_eval,
				device=F_var.device)).permute(2, 0, 1) * args.ntk_std_scale + y_pred
			prob = F_samples.softmax(-1).mean(0)

			loss += F.cross_entropy(prob.log(), y).item() * x.shape[0]
			conf, pred = torch.max(prob, 1)
			acc += (pred == y).float().sum().item()
			num_data += x.shape[0]

			targets.append(y)
			confidences.append(conf)
			predictions.append(pred)

		targets, confidences, predictions = torch.cat(targets), torch.cat(confidences), torch.cat(predictions)
		loss /= num_data
		acc /= num_data
		ece = _ECELoss()(confidences, predictions, targets).item()
	if verbose:
		print("Test results of ELLA: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}, time: {:.2f}s".format(loss, acc, ece, time.time() - t0))
	if return_more:
		return (targets == predictions).float(), confidences
	return loss, acc, ece

def test(test_loader, model, device, args, verbose=True, return_more=False):
	t0 = time.time()

	model.eval()

	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x_batch, y_batch in test_loader:
			x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
			y_pred = model(x_batch)

			loss += F.cross_entropy(y_pred, y_batch).item() * x_batch.shape[0]
			conf, pred = torch.max(y_pred.softmax(-1), 1)
			acc += (pred == y_batch).float().sum().item()
			num_data += x_batch.shape[0]

			targets.append(y_batch)
			confidences.append(conf)
			predictions.append(pred)

		targets, confidences, predictions = torch.cat(targets), torch.cat(confidences), torch.cat(predictions)
		loss /= num_data
		acc /= num_data
		ece = _ECELoss()(confidences, predictions, targets).item()

	if verbose:
		print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}, time: {:.2f}s".format(loss, acc, ece, time.time() - t0))
	if return_more:
		return (targets == predictions).float(), confidences
	return loss, acc, ece


if __name__ == '__main__':
	main()

