from __future__ import print_function

import math
import os
import pickle
from datetime import datetime as dt
import scipy.stats as st

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sys

from data_utils import get_block_samples
from vis_utils import save_map

batch_size = 50
l2_param = 1e-4
epochs = 10
begin_anneal = 50000
decay_rate = 5e-3
min_lr = 1e-5
lr = 0.01
acf = 'sigmoid'
op_type = 'SGD'
momentum = 0.3
in_dim = 52470
output_dim = 5000
denoising = True
tied = True
level = 0.7
if not denoising:
    level = None
pre_loaded = True
bias = True
subjects = 'all'
noise_type = 'masking'
block_size = 2
use_gpu = True

l1_param = 0.1
RHO = 0.5

month = '0{}'.format(dt.now().month) if dt.now().month < 10 else str(dt.now().month)
day = '0{}'.format(dt.now().day) if dt.now().day < 10 else str(dt.now().day)
hour = '0{}'.format(dt.now().hour) if dt.now().hour < 10 else str(dt.now().hour)
minute = '0{}'.format(dt.now().minute) if dt.now().minute < 10 else str(dt.now().minute)
sec = '0{}'.format(dt.now().second) if dt.now().second < 10 else str(dt.now().second)

rho_str = str(RHO)
rho_str = rho_str.replace('.', '')
pct_str = 'none' if not denoising else str(int(level * 100))
output_folder = '/users/nivl/data/autoencoder/hidden/{}{}/rho{}_{}pct'.format(day, month, rho_str, pct_str)
# output_folder = '/users/nivl/data/dnn_results/DEBUG/{}{}{}_{}{}{}_gpu2205'.format(str(dt.now().year), month, day, hour, minute, sec)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class TiedAutoEncoderOffTheShelf(nn.Module):
    def __init__(self, inp, out, weight1):
        super(TiedAutoEncoderOffTheShelf, self).__init__()
        self.encoder = nn.Linear(inp, out, bias=bias)
        self.decoder = nn.Linear(out, inp, bias=bias)

        # tie the weights
        self.encoder.weight.data = weight1.clone()
        nn.init.kaiming_normal_(self.encoder.weight)
        self.decoder.weight.data = self.encoder.weight.data.transpose(0, 1)

    def forward(self, x):
        x = self.encoder(x)
        enc = torch.sigmoid(x)
        x_hat = self.decoder(enc)
        return x_hat, enc


class DAE(nn.Module):
    def __init__(self, input_dim, n_nodes, layer_idx):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, n_nodes), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Linear(n_nodes, input_dim))
        self._initialize_parameters()

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec

    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias, std=0.01)


def plot_dae_results(cost_list, loss_list, batchcost, batchloss, rho_mat, epoch, output_folder):
    plot = plt.figure()
    plt.title('Total loss term\n Final loss term: {0:.3f}'.format(loss_list[-1]))
    plt.plot(loss_list)
    plot.savefig('{}/loss.png'.format(output_folder))
    plt.close()

    plot = plt.figure()
    plt.title('Cost function value\n Final cost term: {0:.3f}'.format(cost_list[-1]))
    plt.plot(cost_list)
    plot.savefig('{}/cost.png'.format(output_folder))
    plt.close()

    plot = plt.figure()
    plt.title('Total loss term per batch\n Final loss term: {0:.3f}'.format(batchloss[-1]))
    plt.plot(batchloss)
    plot.savefig('{}/loss_batch.png'.format(output_folder))
    plt.close()

    plot = plt.figure()
    plt.title('Cost function value per batch\n Final cost term: {0:.3f}'.format(batchcost[-1]))
    plt.plot(batchcost)
    plot.savefig('{}/cost_batch.png'.format(output_folder))
    plt.close()

    plot = plt.figure()
    plt.title('Rho values')
    plt.ylim(0, 1)
    for i in range(rho_mat.shape[1]):
        node_rho_vec = rho_mat[:epoch, i]
        plt.plot(node_rho_vec)
    plot.savefig('{}/rho_epoch.png'.format(output_folder))
    plt.close()


def adjust_learning_rate(optimizer, epoch):
    """If using LR annealing"""
    if begin_anneal == 0:
        learning_rate = lr * 1.0
    elif epoch > begin_anneal:
        prev_lr = optimizer.param_groups[0]['lr']
        learning_rate = max(min_lr, (-decay_rate * epoch + (1 + decay_rate * begin_anneal)) * prev_lr)
    else:
        learning_rate = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def get_noisy_data(data, level):
    [num_samples, in_dim] = data.shape
    num_corrupt = int(level * in_dim)
    for i in range(num_samples):
        corrupt_ids = np.random.choice(np.arange(in_dim), replace=False, size=num_corrupt)
        data[i][corrupt_ids] = 0
    return data


def kl_divergence(rho, rho_hat):
    kldiv = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    sum = torch.sum(kldiv)
    return sum


def train(model, optimizer, all_scans, batchcost, batchloss, cost_list, loss_list, epoch, tied, denoising, total_batches, rho, rho_mat):
    model.train()
    criterion = nn.MSELoss()
    running_loss = []
    running_cost = []
    num_blocks = int(len(all_scans) / block_size)

    for block in range(num_blocks):
        if block % 10 == 0:
            print('Block {}'.format(block))
        block_paths = all_scans[block * block_size: (block + 1) * block_size]
        block_clean = get_block_samples(block_paths)
        if use_gpu:
            block_clean = block_clean.cuda()
        ids = np.arange(block_clean.shape[0])
        block_noisy = get_noisy_data(block_clean.clone(), level) if denoising else None
        np.random.shuffle(ids)
        num_batches = int(math.ceil(block_clean.shape[0] / batch_size))
        for batch_idx in range(num_batches):
            batch_start = batch_size * batch_idx
            batch_end = batch_size * (batch_idx + 1) if batch_idx < (num_batches - 1) else -1
            target = block_clean[batch_start: batch_end]
            data = block_noisy[batch_start: batch_end] if denoising else target
            output, enc_batch = model(data)
            rho_hat = torch.mean(enc_batch, dim=0, keepdim=True)
            sparse_term = kl_divergence(rho, rho_hat)
            cost_val = criterion(output, target)
            loss = cost_val + sparse_term * l1_param
            running_loss.append(loss.item())
            running_cost.append(cost_val.item())
            batchloss.append(loss.item())
            batchcost.append(cost_val.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batches += 1
    if tied:
        rho_mat[epoch - 1] = rho_hat.detach().cpu().numpy().flatten()
    else:
        for i in range(2):
            rho_mat[i][epoch - 1] = rho_hat.detach().cpu().numpy().flatten()

    running_cost = [x for x in running_cost if not np.isnan(x)]
    running_loss = [x for x in running_loss if not np.isnan(x)]
    epoch_loss = np.mean(running_loss)
    epoch_cost = np.mean(running_cost)
    loss_list.append(epoch_loss)
    cost_list.append(epoch_cost)
    print("======> epoch: {}/{}. Avg. rho_hat: {}, Loss: {}, Cost: {}".format(epoch, epochs,np.mean(rho_mat[epoch - 1]), epoch_loss, epoch_cost))

    return cost_list, loss_list, batchcost, batchloss, rho_mat, total_batches


f = open(output_folder + "/parameters.txt", 'w')
f.write('n_epochs : ' + str(epochs) + '\n')
f.write('batch_size : ' + str(batch_size) + '\n')
f.write('init_learning_rate : ' + str(lr) + '\n')
f.write('begin_anneal : ' + str(begin_anneal) + '\n')
f.write('decay_rate : ' + str(decay_rate) + '\n')
f.write('min_lr : ' + str(min_lr) + '\n')
f.write('rho : ' + str(RHO) + '\n')
f.write('L1 param : ' + str(l1_param) + '\n')
f.write('L2_reg : ' + str(l2_param) + '\n')
f.write('input_dim : ' + str(in_dim) + '\n')
f.write('output_dim : ' + str(output_dim) + '\n')
f.write('denoising : ' + str(denoising) + '\n')
f.write('masking level : ' + str(level) + '\n')
f.write('tied : ' + str(tied) + '\n')
f.write('activation function : ' + str(acf) + '\n')
f.write('momentum : ' + str(momentum) + '\n')
f.write('subjects : ' + str(subjects) + '\n')
f.write('bias : ' + str(bias) + '\n')
f.write('noise_type : ' + str(noise_type) + '\n')
f.close()

model_dir = output_folder + '/models'
os.makedirs(model_dir)

pkl = pickle.load(open("all_scans.p", "rb"))
all_scans = pkl['all_scans']

net = TiedAutoEncoderOffTheShelf(in_dim, output_dim, torch.randn(output_dim, in_dim)) if tied else DAE(in_dim,
                                                                                                       output_dim,
                                                                                                       torch.randn(
                                                                                                           output_dim,
                                                                                                           in_dim))
if use_gpu:
    net = net.cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                      weight_decay=l2_param) if op_type is 'SGD' else optim.Adam(net.parameters(), lr=lr,
                                                                                 weight_decay=1e-4)
cost_list = []
loss_list = []
batchcost = []
batchloss = []

batches_per_epoch = int(math.ceil(3896858 / batch_size))
n_batches = epochs * batches_per_epoch

rho_mat = np.zeros((epochs, output_dim)) if tied else [np.zeros((epochs, output_dim)), np.zeros((epochs, output_dim))]

rho = torch.FloatTensor([RHO for _ in range(output_dim)]).unsqueeze(0).cuda()

total_batches = 0
for epoch in range(1, epochs + 1):
    print('Starting epoch {}'.format(epoch))
    adjust_learning_rate(optimizer, epoch)
    cost_list, loss_list, batchcost, batchloss, rho_mat, total_batches = train(net, optimizer, all_scans, batchcost, batchloss,
                                                                               cost_list, loss_list, epoch, tied,
                                                                               denoising, total_batches, rho, rho_mat)
    if epoch % 1 == 0:
        torch.save(net.state_dict(), model_dir + '/epoch_{}.pt'.format(epoch))
        save_map(model_dir, epoch)
        plot_dae_results(cost_list, loss_list, batchcost, batchloss, rho_mat, epoch, output_folder)
