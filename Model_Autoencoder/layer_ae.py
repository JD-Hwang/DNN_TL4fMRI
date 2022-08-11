from __future__ import print_function

import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from datetime import datetime as dt
from data_utils import get_matfile_data
from vis_utils import save_map

batch_size = 10
l2_param = 5e-4
epochs = 1000
begin_anneal = 50000
decay_rate = 1e-4
min_lr = 1e-5
lr = 0.01
acf = 'relu'
op_type = 'SGD'
max_b = 5e-5
b_lr = 5e-6
momentum = 0.3
tg_hsp = 0.5
in_dim = 59583
output_dim = 10000
denoising = True
tied = True
pct = 50
if not denoising:
    pct = None
pre_loaded = False
bias = True
subjects = 10
eps = 1e-3
noise_type = 'masking'


month = '0{}'.format(dt.now().month) if dt.now().month < 10 else str(dt.now().month)
day = '0{}'.format(dt.now().day) if dt.now().day < 10 else str(dt.now().day)
hour = '0{}'.format(dt.now().hour) if dt.now().hour < 10 else str(dt.now().hour)
minute = '0{}'.format(dt.now().minute) if dt.now().minute < 10 else str(dt.now().minute)
sec = '0{}'.format(dt.now().second) if dt.now().second < 10 else str(dt.now().second)

hsp_str = str(tg_hsp)
hsp_str = hsp_str.replace('.', '')
pct_str = 'none' if not denoising else str(pct)
output_folder = '/users/nivl/data/autoencoder/hsp/hsp{}_{}pct2_{}_layer'.format(hsp_str, pct_str, noise_type)
# output_folder = '/users/nivl/data/autoencoder/hsp/{}{}{}_{}{}{}'.format(str(dt.now().year), month, day, hour, minute, sec)
# output_folder = '/users/nivl/data/dnn_results/DEBUG/{}{}{}_{}{}{}'.format(str(dt.now().year), month, day, hour, minute, sec)
#
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
        return reconstructed_output, encoded_feats


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


def plot_dae_results(cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list, nzr_list, dae_dir, tied):
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

    if tied:
        plot = plt.figure()
        plt.title('HSP value: {0:.3f}'.format(sparsity_list[-1]))
        plt.plot(sparsity_list)
        plot.savefig('{}/hsp.png'.format(dae_dir))
        plt.close()

        plot = plt.figure()
        plt.title('Beta value')
        plt.plot(beta_list)
        plot.savefig('{}/beta.png'.format(dae_dir))
        plt.close()

        plot = plt.figure()
        plt.title('NZR value: {0:.4f}'.format(nzr_list[-1]))
        plt.plot(nzr_list)
        plot.savefig('{}/nzr.png'.format(dae_dir))
        plt.close()
    else:
        plot = plt.figure()
        plt.title('HSP value')
        for i in range(2):
            plt.plot(sparsity_list[i], label='layer{}'.format(i+1))
        plt.legend()
        plot.savefig('{}/hsp.png'.format(dae_dir))
        plt.close()

        plot = plt.figure()
        plt.title('Beta value')
        for i in range(2):
            plt.plot(beta_list[i], label='layer{}'.format(i+1))
        plt.legend()
        plot.savefig('{}/beta.png'.format(dae_dir))
        plt.close()

        plot = plt.figure()
        plt.title('NZR value')
        for i in range(2):
            plt.plot(nzr_list[i], label='layer{}'.format(i + 1))
        plt.legend()
        plot.savefig('{}/nzr.png'.format(dae_dir))
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
    """Weight sparsity control with NZR sparsness (Layer wise)"""

    # Get value of weight
    [dim, n_nodes] = w.shape
    num_elements = n_nodes * dim

    # Calculate NZR and HSP sparsness
    norm_ratio = torch.norm(w, 1) / torch.norm(w, 2)
    h = (np.sqrt(num_elements) - norm_ratio.item()) / (np.sqrt(num_elements) - 1)

    wthre = torch.gt(torch.abs(w), eps)
    nzr = torch.nonzero(wthre).size(0) / num_elements

    # Update beta
    b -= b_lr * np.sign(h - tg)

    # Trim value
    b = 0.0 if b < 0.0 else b
    b = max_b if b > max_b else b

    return h, b, nzr


def l1_penalty(model, beta_val, max_b, b_lr, tg_hsp, tied):
    model_layers = [x for x in list(model.parameters()) if len(x.shape) == 2]
    l1_reg = None
    hsp_val = [0] if tied else [0, 0]
    nzr_val = [0] if tied else [0, 0]
    for i in range(2):
        if tied and i == 1:
            continue
        W = model_layers[i]
        layer_maxb = max_b if i == 0 else max_b * 10
        layer_blr = b_lr if i == 0 else b_lr * 10
        hsp_val[i], beta_val[i], nzr_val[i] = hsp_sparsity_control(W, beta_val[i], layer_maxb, layer_blr, tg_hsp)
        layer_l1 = torch.norm(W, 1) * beta_val[i]
        if l1_reg is None:
            l1_reg = layer_l1
        else:
            l1_reg = l1_reg + layer_l1

    return l1_reg, hsp_val, beta_val, nzr_val


def l2_penalty(model):
    l2_reg = None
    model_layers = [x for x in list(model.parameters()) if len(x.shape) == 2]
    for i in range(1):
        layer = model_layers[i]
        if l2_reg is None:
            l2_reg = torch.sum(layer ** 2)
        else:
            l2_reg = l2_reg + torch.sum(layer ** 2)
    return l2_reg


def train(model, optimizer, train_samples, noisy_samples, beta_val, max_b, b_lr, tg_hsp, batchcost, batchloss, cost_list, loss_list, sparsity_list, beta_list, nzr_list, epoch, tied, denoising):
    model.train()
    criterion = nn.MSELoss()
    ids = np.arange(train_samples.shape[0])
    np.random.shuffle(ids)
    start_id = 0
    end_id = batch_size
    running_loss = 0
    running_cost = 0
    while end_id < len(ids):
        end_id = start_id + batch_size
        if end_id > len(ids):
            end_id = len(ids)
        batch_ids = ids[start_id:end_id]
        data = noisy_samples[batch_ids] if denoising else train_samples[batch_ids]
        target = train_samples[batch_ids]
        output, _ = model(data)
        cost_val = criterion(output, target)
        # cost_val = torch.sum(torch.sum((output - target) ** 2, 0)) / batch_size
        l1_term, hsp_val, beta_val, nzr_val = l1_penalty(model, beta_val, max_b, b_lr, tg_hsp, tied)
        # l2_term = l2_penalty(model) * l2_param
        loss = cost_val + l1_term# + l2_term
        running_loss += loss.item()
        running_cost += cost_val.item()
        batchloss.append(loss.item())
        batchcost.append(cost_val.item())
        if tied:
            sparsity_list.extend(hsp_val)
            beta_list.extend(beta_val)
            nzr_list.extend(nzr_val)
        else:
            for i in range(2):
                sparsity_list[i].append(hsp_val[i])
                beta_list[i].append(beta_val[i])
                nzr_list[i].append(nzr_val[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        start_id += batch_size

    epoch_loss = running_loss / (len(train_samples) / batch_size)
    epoch_cost = running_cost / (len(train_samples) / batch_size)
    loss_list.append(epoch_loss)
    cost_list.append(epoch_cost)
    print("======> epoch: {}/{}, HSP: {}, Beta: {}, Loss: {}, Cost: {}".format(epoch, epochs, hsp_val, beta_val, epoch_loss, epoch_cost))

    return cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list, nzr_list


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
f.write('masking level : ' + str(pct) + '\n')
f.write('tied : ' + str(tied) + '\n')
f.write('activation function : ' + str(acf) + '\n')
f.write('momentum : ' + str(momentum) + '\n')
f.write('subjects : ' + str(subjects) + '\n')
f.write('bias : ' + str(bias) + '\n')
f.write('noise_type : ' + str(noise_type) + '\n')
f.close()

model_dir = output_folder + '/models'
os.makedirs(model_dir)

samples, noisy_samples = get_matfile_data(subjects, in_dim, pre_loaded, denoising, pct, noise_type)

samples = samples.cuda()
if noisy_samples is not None:
    noisy_samples = noisy_samples.cuda()

net = TiedAutoEncoderOffTheShelf(in_dim, output_dim, torch.randn(output_dim, in_dim)) if tied else DAE(in_dim, output_dim, torch.randn(output_dim, in_dim))
net = net.cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=l2_param) if op_type is 'SGD' else optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
beta_val = [0] if tied else [0, 0]
cost_list = []
loss_list = []
batchcost = []
batchloss = []
sparsity_list = [] if tied else [[], []]
beta_list = [] if tied else [[], []]
nzr_list = [] if tied else [[], []]
for epoch in range(1, epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list, nzr_list = train(net, optimizer, samples,
                                                                                             noisy_samples, beta_val,
                                                                                             max_b, b_lr, tg_hsp,
                                                                                             batchcost, batchloss,
                                                                                             cost_list, loss_list,
                                                                                             sparsity_list, beta_list, nzr_list,
                                                                                             epoch, tied, denoising)
    if epoch % 50 == 0:
        torch.save(net.state_dict(), model_dir + '/epoch_{}.pt'.format(epoch))
        save_map(model_dir, epoch)
        plot_dae_results(cost_list, loss_list, batchcost, batchloss, sparsity_list, beta_list, nzr_list, output_folder, tied)

