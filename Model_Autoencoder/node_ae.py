from __future__ import print_function

import collections
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from datetime import datetime as dt
from data_utils import collect_sbj_data, get_sbj_task_data, is_subject_valid
from vis_utils import save_map
import scipy.stats as stats

batch_size = 50
l2_param = 1e-3
epochs = 100
begin_anneal = 50000
decay_rate = 5e-3
min_lr = 1e-5
lr = 0.01
acf = 'relu'
op_type = 'SGD'
max_b = 5e-2
b_lr = 5e-3
momentum = 0.3
tg_hsp = 0.8
in_dim = 52470
output_dim = 1000
denoising = True
tied = True
level = 0.3
if not denoising:
    level = None
pre_loaded = True
bias = True
subjects = 300
noise_type = 'masking'
block_size = 3000

task = 'MOTOR'


month = '0{}'.format(dt.now().month) if dt.now().month < 10 else str(dt.now().month)
day = '0{}_'.format(dt.now().day) if dt.now().day < 10 else str(dt.now().day)
hour = '0{}'.format(dt.now().hour) if dt.now().hour < 10 else str(dt.now().hour)
minute = '0{}'.format(dt.now().minute) if dt.now().minute < 10 else str(dt.now().minute)
sec = '0{}'.format(dt.now().second) if dt.now().second < 10 else str(dt.now().second)

hsp_str = str(tg_hsp)
hsp_str = hsp_str.replace('.', '')
pct_str = 'none' if not denoising else str(int(level * 100))
output_folder = '/users/nivl/data/autoencoder/hsp/{}{}/hsp{}_{}pct_node1'.format(day, month, hsp_str, pct_str)
# output_folder = '/users/nivl/data/autoencoder/hsp/{}{}{}_{}{}{}'.format(str(dt.now().year), month, day, hour, minute, sec)
# output_folder = '/users/nivl/data/dnn_results/DEBUG/{}{}{}_{}{}{}'.format(str(dt.now().year), month, day, hour, minute, sec)

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
        encoded_feats = self.encoder(x)
        if acf == 'relu':
            encoded_feats = F.relu(encoded_feats, inplace=True)
        elif acf == 'tanh':
            encoded_feats = torch.tanh(encoded_feats)
        else:
            encoded_feats = torch.sigmoid(encoded_feats)
        reconstructed_output = self.decoder(encoded_feats)
        return reconstructed_output


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


def plot_dae_results(cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list, dae_dir, tied, batches_per_epoch):
    plot = plt.figure()
    plt.title('Total loss term\n Final loss term: {0:.3f}'.format(loss_list[-1]))
    plt.plot(loss_list)
    plot.savefig('{}/loss.png'.format(dae_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Cost function value\n Final cost term: {0:.3f}'.format(cost_list[-1]))
    plt.plot(cost_list)
    plot.savefig('{}/cost.png'.format(dae_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Total loss term per batch\n Final loss term: {0:.3f}'.format(batchloss[-1]))
    plt.plot(batchloss)
    plot.savefig('{}/loss_batch.png'.format(dae_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Cost function value per batch\n Final cost term: {0:.3f}'.format(batchcost[-1]))
    plt.plot(batchcost)
    plot.savefig('{}/cost_batch.png'.format(dae_dir))
    plt.close()

    loc = epoch * batches_per_epoch
    if tied:
        plot = plt.figure()
        plt.title('HSP value')
        for i in range(sparsity_list.shape[1]):
            node_hsp_vec = sparsity_list[:loc, i]
            plt.plot(node_hsp_vec)
        plot.savefig('{}/hsp_batch.png'.format(dae_dir))
        plt.close()

        sparsity_epoch = sparsity_list[0::batches_per_epoch]
        plot = plt.figure()
        plt.title('HSP value')
        for i in range(sparsity_list.shape[1]):
            node_hsp_vec = sparsity_epoch[:epoch, i]
            plt.plot(node_hsp_vec)
        plot.savefig('{}/hsp_epoch.png'.format(dae_dir))
        plt.close()

        plot = plt.figure()
        plt.title('Beta value')
        for i in range(beta_list.shape[1]):
            node_beta_vec = beta_list[:loc, i]
            plt.plot(node_beta_vec)
        plot.savefig('{}/beta_batch.png'.format(dae_dir))
        plt.close()

        beta_epoch = beta_list[0::batches_per_epoch]
        plot = plt.figure()
        plt.title('Beta value')
        for i in range(beta_list.shape[1]):
            node_beta_vec = beta_epoch[:epoch, i]
            plt.plot(node_beta_vec)
        plot.savefig('{}/beta_epoch.png'.format(dae_dir))
        plt.close()

    else:
        for j in range(2):
            plot = plt.figure()
            plt.title('HSP value')
            for i in range(sparsity_list.shape[1]):
                node_hsp_vec = sparsity_list[j][:loc, i]
                plt.plot(node_hsp_vec)
            plt.legend()
            plot.savefig('{}/hsp_{}_batch.png'.format(dae_dir, j))
            plt.close()

        for j in range(2):
            plot = plt.figure()
            plt.title('Beta value')
            for i in range(beta_list.shape[1]):
                node_beta_vec = beta_list[j][:loc, i]
                plt.plot(node_beta_vec)
            plt.legend()
            plot.savefig('{}/beta_{}_batch.png'.format(dae_dir, j))
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


def hsp_sparsity_control(w, b, max_b, b_lr, tg):
    """Weight sparsity control with HSP (Node wise)"""

    # Get value of weight
    [dim, n_nodes] = w.shape
    # Calculate HSP sparsness
    norm_ratio = torch.norm(w, 1, 1) / torch.norm(w, 2, 1)
    h = (math.sqrt(n_nodes) - norm_ratio) / (math.sqrt(n_nodes) - 1)

    # Update beta
    tg_vec = torch.from_numpy(np.ones((dim)) * tg).type(torch.FloatTensor).cuda()
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
    for i in range(2):
        if tied and i == 1:
            continue
        W = model_layers[i]
        layer_maxb = max_b if i == 0 else max_b * 10
        layer_blr = b_lr if i == 0 else b_lr * 10
        hsp_val[i], beta_val[i] = hsp_sparsity_control(W, beta_val[i], layer_maxb, layer_blr, tg_hsp)
        layer_l1 = torch.sum(torch.abs(torch.t(W)) * beta_val[i])
        if l1_reg is None:
            l1_reg = layer_l1
        else:
            l1_reg = l1_reg + layer_l1

    return l1_reg, hsp_val, beta_val


def get_noisy_data(data, level):
    [num_samples, in_dim] = data.shape
    num_corrupt = int(level * in_dim)
    for i in range(num_samples):
        corrupt_ids = np.random.choice(np.arange(in_dim), replace=False, size=num_corrupt)
        data[i][corrupt_ids] = 0
    return data


def train(model, optimizer, train_samples, beta_val, max_b, b_lr, tg_hsp, batchcost, batchloss, cost_list, loss_list, sparsity_list, beta_list, epoch, tied, denoising):
    model.train()
    criterion = nn.MSELoss()
    ids = np.arange(train_samples.shape[0])
    np.random.shuffle(ids)
    running_loss = 0
    running_cost = 0
    num_blocks = int(math.ceil(train_samples.shape[0] / block_size))
    num_batches = int(block_size / batch_size)
    batch_ovr = 0
    for block in range(num_blocks):
        block_start = block_size * block
        block_end = block_size * (block + 1) if block < (num_blocks - 1) else -1
        block_ids = ids[block_start: block_end]
        block_clean = train_samples[block_ids].cuda()
        block_noisy = get_noisy_data(block_clean.clone(), level) if denoising else None

        for batch_idx in range(num_batches):
            batch_start = batch_size * batch_idx
            batch_end = batch_size * (batch_idx + 1) if batch_idx < (num_batches - 1) else -1
            target = block_clean[batch_start: batch_end]
            data = block_noisy[batch_start: batch_end] if denoising else target
            output = model(data)
            cost_val = criterion(output, target)
            l1_term, hsp_val, beta_val = l1_penalty(model, beta_val, max_b, b_lr, tg_hsp, tied)
            loss = cost_val + l1_term
            running_loss += loss.item()
            running_cost += cost_val.item()
            batchloss.append(loss.item())
            batchcost.append(cost_val.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loc = int((epoch - 1) * math.ceil(samples.shape[0] / batch_size)) + batch_ovr

            if tied:
                sparsity_list[batch_loc] = torch.clone(hsp_val[0]).detach().cpu().numpy()
                beta_list[batch_loc] = torch.clone(beta_val[0]).detach().cpu().numpy()
            else:
                for i in range(2):
                    sparsity_list[i][batch_loc] = hsp_val[i].data.cpu().numpy()
                    beta_list[i][batch_loc] = beta_val[i].data.cpu().numpy()
            batch_ovr += 1

    epoch_loss = running_loss / (len(train_samples) / batch_size)
    epoch_cost = running_cost / (len(train_samples) / batch_size)
    loss_list.append(epoch_loss)
    cost_list.append(epoch_cost)
    print("======> epoch: {}/{}. Avg. HSP: {}, Loss: {}, Cost: {}".format(epoch, epochs, np.mean(sparsity_list[epoch - 1]), epoch_loss, epoch_cost))

    return cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list


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
os.makedirs(model_dir)
print(str(dt.now()))
# samples = collect_sbj_data(subjects)
samples = None
sbj_idx = 0
sbj_loaded = 0

hcp_sbj = [e for e in os.listdir('/data4/open_data/HCP') if e.isdecimal()]
random.shuffle(hcp_sbj)

while sbj_loaded < subjects:
    if not is_subject_valid(hcp_sbj[sbj_idx], task):
        sbj_idx += 1
        continue
    sbj_samples, sbj_labels = get_sbj_task_data(hcp_sbj[sbj_idx], task)
    if samples is None:
        samples = sbj_samples
    else:
        samples = np.concatenate((samples, sbj_samples))
    sbj_idx += 1
    sbj_loaded += 1
    print(sbj_loaded)

samples = stats.zscore(samples, axis=0, ddof=1)
samples = torch.from_numpy(samples)
print(samples.shape)
print('loaded data')
print(str(dt.now()))

batches_per_epoch = int(math.ceil(samples.shape[0] / batch_size))
n_batches = epochs * batches_per_epoch

net = TiedAutoEncoderOffTheShelf(in_dim, output_dim, torch.randn(output_dim, in_dim)) if tied else DAE(in_dim, output_dim, torch.randn(output_dim, in_dim))
net = net.cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=l2_param) if op_type is 'SGD' else optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
beta_val = [None] if tied else [None, None]
cost_list = []
loss_list = []
batchcost = []
batchloss = []
sparsity_list = np.zeros((n_batches, output_dim)) if tied else [np.zeros((n_batches, output_dim)), np.zeros((n_batches, output_dim))]
beta_list = np.zeros((n_batches, output_dim)) if tied else [np.zeros((n_batches, output_dim)), np.zeros((n_batches, output_dim))]
for epoch in range(1, epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list = train(net, optimizer, samples, beta_val,
                                                                                             max_b, b_lr, tg_hsp,
                                                                                             batchcost, batchloss,
                                                                                             cost_list, loss_list,
                                                                                             sparsity_list, beta_list,
                                                                                             epoch, tied, denoising)
    if epoch % 10 == 0:
        torch.save(net.state_dict(), model_dir + '/epoch_{}.pt'.format(epoch))
        save_map(model_dir, epoch)
        plot_dae_results(cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list, output_folder, tied, batches_per_epoch)

