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
l2_param = 1e-3
epochs = 20
begin_anneal = 50000
decay_rate = 5e-3
min_lr = 1e-5
lr = 0.01
acf = 'relu'
op_type = 'SGD'
max_b = 5e-5
b_lr = 5e-6
momentum = 0.3
tg_hsp = 0.8
in_dim = 52470
output_dim = 5000
denoising = False
tied = True
level = 0
if not denoising:
    level = None
pre_loaded = True
bias = True
subjects = 'all'
noise_type = 'masking'
block_size = 2
use_gpu = True
scan = 'REST2'
phase = 'RL'
resume = False
latest_epoch = 0

resumed_path = '/users/nivl/data/autoencoder/hsp/rep/1709'

month = '0{}'.format(dt.now().month) if dt.now().month < 10 else str(dt.now().month)
day = '0{}'.format(dt.now().day) if dt.now().day < 10 else str(dt.now().day)
hour = '0{}'.format(dt.now().hour) if dt.now().hour < 10 else str(dt.now().hour)
minute = '0{}'.format(dt.now().minute) if dt.now().minute < 10 else str(dt.now().minute)
sec = '0{}'.format(dt.now().second) if dt.now().second < 10 else str(dt.now().second)

hsp_str = str(tg_hsp)
hsp_str = hsp_str.replace('.', '')
pct_str = 'none' if not denoising else str(int(level * 100))
output_folder = '/users/nivl/data/autoencoder/hsp/rep/{}{}/hsp{}_{}pct_{}_{}'.format(day, month, hsp_str, pct_str, scan, phase)
# output_folder = '/users/nivl/data/autoencoder/hsp/{}{}{}_{}{}{}'.format(str(dt.now().year), month, day, hour, minute, sec)


if resume is True:
    run_path = [x for x in os.listdir(resumed_path) if all(sub in x for sub in [scan, phase])]
    output_folder = '{}/{}'.format(resumed_path, run_path[0])
    past_epochs = [int(x[6:-3]) for x in os.listdir(output_folder + '/models') if x.endswith('.pt')]
    latest_epoch = max(past_epochs)
    model_dict = torch.load('{}/models/epoch_{}.pt'.format(output_folder, latest_epoch))
    print('Resuming from epoch {}'.format(latest_epoch))
    print(output_folder)

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
        if acf == 'relu':
            x = F.relu(x, inplace=True)
        elif acf == 'tanh':
            x = torch.tanh(x)
        else:
            x = torch.sigmoid(x)
        x = self.decoder(x)
        return x


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


def plot_dae_results(cost_list, loss_list, batchcost, batchloss, sparsity_batch, beta_batch, sparsity_epoch, beta_epoch,
                     epoch, output_folder, tied, total_batches):
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

    if tied:
        plot = plt.figure()
        plt.title('HSP value')
        plt.ylim(0, 1)
        for i in range(sparsity_epoch.shape[1]):
            node_hsp_vec = sparsity_epoch[:epoch, i]
            plt.plot(node_hsp_vec)
        plot.savefig('{}/hsp_epoch.png'.format(output_folder))
        plt.close()

        plot = plt.figure()
        plt.title('Beta value')
        for i in range(beta_epoch.shape[1]):
            node_beta_vec = beta_epoch[:epoch, i]
            plt.plot(node_beta_vec)
        plot.savefig('{}/beta_epoch.png'.format(output_folder))
        plt.close()


def save_epoch_results(cost_list, loss_list, batchcost, batchloss, sparsity_epoch, beta_epoch):
    res_dict = {'cost_list': cost_list, 'loss_list': loss_list, 'batchcost': batchcost, 'batchloss': batchloss,
                'sparsity_epoch': sparsity_epoch, 'beta_epoch': beta_epoch}
    pickle.dump(res_dict, open(output_folder + '/res_dict.p', "wb"))


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


def hsp_sparsity_control(w, b, max_b, b_lr, tg):
    """Weight sparsity control with HSP (Node wise)"""

    # Get value of weight
    [dim, n_nodes] = w.shape
    # Calculate HSP sparsness
    norm_ratio = torch.norm(w, 1, 1) / torch.norm(w, 2, 1)
    h = (math.sqrt(n_nodes) - norm_ratio) / (math.sqrt(n_nodes) - 1)

    # Update beta
    tg_vec = torch.from_numpy(np.ones((dim)) * tg).type(torch.FloatTensor)
    if use_gpu:
        tg_vec = tg_vec.cuda()
    diff = h - tg_vec

    if b is None:
        b = -1 * b_lr * torch.sign(diff)
    else:
        b.detach_()
        b -= b_lr * torch.sign(diff)

    # Trim value
    b[b < 0.0] = 0.0
    b[b > max_b] = max_b

    return h, b


def l1_penalty(model, beta_val, max_b, b_lr, tg_hsp, tied):
    model_layers = [x for x in list(model.parameters()) if len(x.shape) == 2]
    l1_reg = None
    hsp_val = [None] if tied else [None, None]
    w = model_layers[0]
    layer_maxb = max_b
    layer_blr = b_lr
    hsp_val[0], beta_val[0] = hsp_sparsity_control(w, beta_val[0], layer_maxb, layer_blr, tg_hsp)

    layer_l1 = torch.sum(torch.abs(torch.t(w)) * beta_val[0])

    return layer_l1, hsp_val, beta_val


def get_noisy_data(data, level):
    [num_samples, in_dim] = data.shape
    num_corrupt = int(level * in_dim)
    for i in range(num_samples):
        corrupt_ids = np.random.choice(np.arange(in_dim), replace=False, size=num_corrupt)
        data[i][corrupt_ids] = 0
    return data


def train(model, optimizer, all_scans, beta_val, max_b, b_lr, tg_hsp, batchcost, batchloss, cost_list, loss_list,
          sparsity_batch, beta_batch, sparsity_epoch, beta_epoch, epoch, tied, denoising, total_batches):
    model.train()
    criterion = nn.MSELoss()
    running_loss = []
    running_cost = []
    num_blocks = int(len(all_scans) / block_size)

    for block in range(num_blocks - 1):
        if block % 10 == 0:
            print('Block {}/{}'.format(block, num_blocks))
            print('nan cost: {}'.format(np.argwhere(np.isnan(running_cost))))
            print('nan loss: {}'.format(np.argwhere(np.isnan(running_loss))))
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
            output = model(data)
            cost_val = criterion(output, target)
            l1_term, hsp_val, beta_val = l1_penalty(model, beta_val, max_b, b_lr, tg_hsp, tied)
            loss = cost_val + l1_term
            running_loss.append(loss.item())
            running_cost.append(cost_val.item())

            batchloss.append(loss.item())
            batchcost.append(cost_val.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batches += 1

    if tied:
        sparsity_epoch[epoch - 1] = torch.clone(hsp_val[0]).detach().cpu().numpy()
        beta_epoch[epoch - 1] = torch.clone(beta_val[0]).detach().cpu().numpy()
    else:
        for i in range(2):
            sparsity_epoch[i][epoch - 1] = hsp_val[i].data.cpu().numpy()
            beta_epoch[i][epoch - 1] = beta_val[i].data.cpu().numpy()

    print('nan cost: {}'.format(np.argwhere(np.isnan(running_cost))))
    print('nan loss: {}'.format(np.argwhere(np.isnan(running_loss))))
    running_cost = [x for x in running_cost if not np.isnan(x)]
    running_loss = [x for x in running_loss if not np.isnan(x)]
    epoch_loss = np.mean(running_loss)
    epoch_cost = np.mean(running_cost)
    loss_list.append(epoch_loss)
    cost_list.append(epoch_cost)
    print("======> epoch: {}/{}. Avg. HSP: {}, Loss: {}, Cost: {}".format(epoch, epochs,
                                                                          np.mean(sparsity_epoch[epoch - 1]),
                                                                          epoch_loss, epoch_cost))

    return cost_list, loss_list, batchcost, batchloss, sparsity_batch, beta_batch, sparsity_epoch, beta_epoch, total_batches


f = open(output_folder + "/parameters.txt", 'w')
f.write('n_epochs : ' + str(epochs) + '\n')
f.write('batch_size : ' + str(batch_size) + '\n')
f.write('init_learning_rate : ' + str(lr) + '\n')
f.write('begin_anneal : ' + str(begin_anneal) + '\n')
f.write('decay_rate : ' + str(decay_rate) + '\n')
f.write('min_lr : ' + str(min_lr) + '\n')
f.write('beta_lrate : ' + str(b_lr) + '\n')
f.write('L2_reg : ' + str(l2_param) + '\n')
f.write('max_beta : ' + str(max_b) + '\n')
f.write('input_dim : ' + str(in_dim) + '\n')
f.write('output_dim : ' + str(output_dim) + '\n')
f.write('tg_hsp : ' + str(tg_hsp) + '\n')
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
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# print(str(dt.now()))
# samples = collect_sbj_data(subjects)
# print('loaded data')
# print(str(dt.now()))

pkl = pickle.load(open("all_scans.p", "rb"))
all_scans = pkl['all_scans']

run_scans = [x for x in all_scans if all(sub in x for sub in [scan, phase])]
net = TiedAutoEncoderOffTheShelf(in_dim, output_dim, torch.randn(output_dim, in_dim)) if tied else DAE(in_dim,
                                                                                                       output_dim,
                                                                                                       torch.randn(
                                                                                                           output_dim,
                                                                                                           in_dim))
if use_gpu:
    net = net.cuda()

if resume is True:
    net.load_state_dict(model_dict)

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                      weight_decay=l2_param) if op_type is 'SGD' else optim.Adam(net.parameters(), lr=lr,
                                                                                 weight_decay=1e-4)
beta_val = [None] if tied else [None, None]
cost_list = []
loss_list = []
batchcost = []
batchloss = []

sparsity_batch = None
beta_batch = None
sparsity_epoch = np.zeros((epochs, output_dim)) if tied else [np.zeros((epochs, output_dim)),
                                                              np.zeros((epochs, output_dim))]
beta_epoch = np.zeros((epochs, output_dim)) if tied else [np.zeros((epochs, output_dim)),
                                                          np.zeros((epochs, output_dim))]

total_batches = 0
for epoch in range(latest_epoch + 1, epochs + 1):
    print('Starting epoch {}'.format(epoch))
    adjust_learning_rate(optimizer, epoch)
    cost_list, loss_list, batchcost, batchloss, sparsity_batch, beta_batch, sparsity_epoch, beta_epoch, total_batches = train(
        net, optimizer, all_scans, beta_val,
        max_b, b_lr, tg_hsp,
        batchcost, batchloss,
        cost_list, loss_list,
        sparsity_batch, beta_batch, sparsity_epoch, beta_epoch,
        epoch, tied, denoising, total_batches)
    if epoch % 1 == 0:
        torch.save(net.state_dict(), model_dir + '/epoch_{}.pt'.format(epoch))
        save_map(model_dir, epoch)
        plot_dae_results(cost_list, loss_list, batchcost, batchloss, sparsity_batch, beta_batch, sparsity_epoch,
                         beta_epoch, epoch, output_folder, tied, total_batches)
        save_epoch_results(cost_list, loss_list, batchcost, batchloss, sparsity_epoch, beta_epoch)
