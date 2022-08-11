from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from datetime import datetime as dt
import random
from data_utils import get_sbj_task_data, is_subject_valid, get_task_retest_data, convert_test_data
import timeit
import itertools
import collections
import scipy.io as sio
from numpy import linalg as LA
import pickle


tasks = {'WM': 8, 'MOTOR': 5, 'EMOTION': 2, 'RELATIONAL': 2, 'SOCIAL': 2, 'LANGUAGE': 2, 'GAMBLING': 2}
task_lst = ['WM', 'MOTOR', 'EMOTION', 'RELATIONAL', 'SOCIAL', 'LANGUAGE', 'GAMBLING']
num_samples = [[544], [260], [223, 224], [216], [268], [300], [271]]

retest_sbj = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', '137128',
              '139839', '144226', '146129', '149337', '149741', '151526', '158035', '169343', '172332',
              '175439', '177746', '185442', '187547', '192439', '194140', '195041', '200109', '200614',
              '204521', '250427', '287248', '433839', '562345', '599671', '601127', '627549', '660951',
              '662551', '783462', '859671', '861456', '877168', '917255']

batch_size = 50
lr = 1e-3
begin_anneal = 75
decay_rate = 2e-3
min_lr = 1e-4
op_type = 'SGD'
momentum = 0.3
ae_hsp = 0.8
in_dim = 52470
hidden_dim = 5000
level = 0
task_id = 0
task = task_lst[task_id]
output_dim = tasks[task]
samples_per_sbj = num_samples[task_id]
l1_candidates = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
l2_candidates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
tg_lambda = list(itertools.product(l1_candidates, l2_candidates))
tg_lambda = [list(i) for i in tg_lambda]
layers_dim = [1000]
model = 'dbn'
print(model)

month = '0{}'.format(dt.now().month) if dt.now().month < 10 else str(dt.now().month)
day = '0{}'.format(dt.now().day) if dt.now().day < 10 else str(dt.now().day)
hour = '0{}'.format(dt.now().hour) if dt.now().hour < 10 else str(dt.now().hour)
minute = '0{}'.format(dt.now().minute) if dt.now().minute < 10 else str(dt.now().minute)
sec = '0{}'.format(dt.now().second) if dt.now().second < 10 else str(dt.now().second)


hsp_str = str(ae_hsp)
hsp_str = hsp_str.replace('.', '')
pct_str = str(int(level * 100))
print(hsp_str)
print(pct_str)

output_folder = '/users/nivl/data/autoencoder/classifier/fixed/{}{}/{}{}{}_{}_hsp{}_pct{}_{}'.format(month, day, hour, minute, sec, task, hsp_str, pct_str, model)
# output_folder = '/users/nivl/data/dnn_results/DEBUG/{}{}{}_{}{}{}'.format(str(dt.now().year), month, day, hour, minute, sec)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

f = open(output_folder + "/parameters.txt", 'w')
f.write('optimizer_algorithm : ' + str(op_type) + '\n')
f.write('batch_size : ' + str(batch_size) + '\n')
f.write('init_learning_rate : '+str(lr)+'\n')
f.write('begin_anneal : ' + str(begin_anneal) + '\n')
f.write('decay_rate : ' + str(decay_rate) + '\n')
f.write('min_lr : ' + str(min_lr) + '\n')
f.write('L2_reg : ' + str(l2_candidates) + '\n')
f.write('L1_reg : ' + str(l1_candidates) + '\n')
f.close()


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes):
        super(DNN, self).__init__()
        self.layers = self.make_layers(input_dim, hidden_layers, num_classes)
        self._initialize_parameters()

    def forward(self, x):
        x = self.layers(x)
        x = F.log_softmax(x, dim=1)
        return x

    def make_layers(self, input_dim, hidden_layers, num_classes):
        layers = [('fc1', nn.Linear(input_dim, hidden_layers[0])), ('relu1', nn.ReLU())]
        in_dim = hidden_layers[0]

        for i in range(1, len(hidden_layers)):
            fc_layer_name = 'fc{}'.format(i + 1)
            relu_layer_name = 'relu{}'.format(i + 1)
            layers += [(fc_layer_name, nn.Linear(in_dim, hidden_layers[i])), (relu_layer_name, nn.ReLU())]
            in_dim = hidden_layers[i]

        layers += [('output', nn.Linear(in_dim, num_classes))]
        layers_dict = collections.OrderedDict(layers)
        return nn.Sequential(layers_dict)

    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias, std=0.01)


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.encoder(x), inplace=True)
        return x


hcp_sbj = [e for e in os.listdir('/data4/open_data/HCP') if e.isdecimal()]
# random.shuffle(hcp_sbj)
enc = Encoder(in_dim, hidden_dim).cuda()
if model == 'ae':
    pretrained_dict = torch.load('/users/nivl/data/autoencoder/hsp/all/hsp{}_{}pct/models/epoch_10.pt'.format(hsp_str, pct_str))
    enc_dict = enc.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in enc_dict}
    # 2. overwrite entries in the existing state dict
    enc_dict.update(pretrained_dict)
    # 3. load the new state dict
    enc.load_state_dict(enc_dict)
else:
    mat = sio.loadmat('/users/nivl/data/autoencoder/dbn/epoch15/1_15epoc.mat')
    weight = torch.from_numpy(mat['tmp_w'])
    bias = torch.flatten(torch.from_numpy(mat['tmp_c']))
    enc.encoder.weight.data = weight
    enc.encoder.bias.data = bias
    enc = enc.cuda()

for param in enc.parameters():
    param.requires_grad = False

train_samples = None
train_labels = None

test_samples = None
test_labels = None

sbj_idx = 0
sbj_loaded = 0

for sbj in hcp_sbj:
    # if sbj_loaded > 20:
    #     continue
    if not is_subject_valid(sbj, task, False):
        continue
    sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)
    if len(sbj_labels) not in samples_per_sbj and task_id != 5:
        print('bad sbj')
        print(len(sbj_labels))
        continue
    if len(sbj_labels) == 224 and task_id == 2:
        sbj_samples = sbj_samples[0:-1]
        sbj_labels = sbj_labels[0:-1]

    sbj_samples = torch.from_numpy(sbj_samples).cuda()
    enc_output = enc(sbj_samples).cpu()
    sbj_samples = sbj_samples.cpu()
    if train_samples is None:
        train_samples = enc_output
        train_labels = sbj_labels
    else:
        train_samples = torch.cat((train_samples, enc_output))
        train_labels = np.concatenate((train_labels, sbj_labels))
    sbj_loaded += 1
    if (sbj_loaded - 1) % 50 == 0:
        print(sbj_loaded)

print('loaded {} training subjects'.format(sbj_loaded))

sbj_loaded = 0
for sbj in retest_sbj:
    # if sbj_loaded > 5:
    #     continue
    if not is_subject_valid(sbj, task, True):
        continue
    sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)
    if len(sbj_labels) not in samples_per_sbj and task_id != 5:
        continue
    if len(sbj_labels) == 224 and task_id == 2:
        sbj_samples = sbj_samples[0:-1]
        sbj_labels = sbj_labels[0:-1]
    sbj_samples = torch.from_numpy(sbj_samples).cuda()
    enc_output = enc(sbj_samples).cpu()
    sbj_samples = sbj_samples.cpu()
    if test_samples is None:
        test_samples = enc_output
        test_labels = sbj_labels
    else:
        test_samples = torch.cat((test_samples, enc_output))
        test_labels = np.concatenate((test_labels, sbj_labels))
    sbj_loaded += 1

print("loaded test data")

train_labels = torch.from_numpy(train_labels)
test_labels = torch.from_numpy(test_labels)


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


def plot_results(cost_list, loss_list, batchcost, batchloss, train_accuracies, test_accuracies, batch_accuracies, lr_list, batch_val, weight_dict, out_dir):
    plot = plt.figure()
    plt.title('Train/Test Accuracies\n Train acc.: {0:.3f}, Test acc. :{1:.3f}'.format(train_accuracies[-1], test_accuracies[-1]))
    plt.plot(train_accuracies, '-b', label="Training acc.")
    plt.plot(test_accuracies, '-r', label="Test acc.")
    plt.ylim(0, 105)
    plt.legend()
    plot.savefig('{}/accuracies.png'.format(out_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Train Accuracy per mini batch\n Train acc.: {0:.3f}'.format(batch_accuracies[-1]))
    plt.plot(batch_accuracies, '-b')
    plt.ylim(0, 105)
    plot.savefig('{}/batch_accuracy.png'.format(out_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Learning rate. Final LR value {0:.3f}'.format(lr_list[-1]))
    plt.plot(lr_list)
    plot.savefig('{}/lr.png'.format(out_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Total loss term\n Final loss term: {0:.3f}'.format(loss_list[-1]))
    plt.plot(loss_list)
    plot.savefig('{}/loss.png'.format(out_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Cost function value\n Final cost term: {0:.3f}'.format(cost_list[-1]))
    plt.plot(cost_list)
    plot.savefig('{}/cost.png'.format(out_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Total loss term per batch\n Final loss term: {0:.3f}'.format(batchloss[-1]))
    plt.plot(batchloss)
    plot.savefig('{}/loss_batch.png'.format(out_dir))
    plt.close()

    plot = plt.figure()
    plt.title('Cost function value per batch\n Final cost term: {0:.3f}'.format(batchcost[-1]))
    plt.plot(batchcost)
    plot.savefig('{}/cost_batch.png'.format(out_dir))
    plt.close()

    for b in range(len(batch_val)):
        plot = plt.figure()
        plt.title('Validation accuracy, batch {}'.format(b + 1))
        plt.plot(batch_val[b])
        plt.ylim(0, 105)
        plot.savefig('{}/batch_val/batch_{}.png'.format(out_dir, b))
        plt.close()

    plot = plt.figure()
    plt.title('Validation accuracy per batch')
    for b in range(len(batch_val)):
        plt.plot(batch_val[b])
    plt.ylim(0, 105)
    plot.savefig('{}/batch_acc.png'.format(out_dir))
    plt.close()

    for key in weight_dict.keys():
        l_name = key[7:].replace('.', '_')
        plot = plt.figure()
        plt.plot(weight_dict[key])
        plot.savefig('{}/{}.png'.format(out_dir, l_name))
        plt.close()


def calc_weight_update(prev, curr, res_dict):
    for key in prev.keys():
        l_prev = prev[key]
        l_curr = curr[key]
        res = LA.norm(l_prev - l_curr) / LA.norm(l_curr)
        if key not in res_dict.keys():
            res_dict[key] = [res]
        else:
            res_dict[key].append(res)
    return res_dict


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


def l1_penalty(net):
    net_layers = [x for x in list(net.parameters()) if len(x.shape) == 2]
    l1_reg = None
    for layer in range(1):
        W = net_layers[layer]
        layer_l1 = torch.norm(W, 1)
        if l1_reg is None:
            l1_reg = layer_l1
        else:
            l1_reg = l1_reg + layer_l1

    return l1_reg


def train(net, optimizer, train_loader, l1_param):
    net.train()
    criterion = nn.NLLLoss()
    running_loss = 0
    running_cost = 0
    correct = 0
    total = 0
    batch_cost = []
    batch_loss = []
    acc_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx + 1 not in [epoch % batches_per_epoch, batches_per_epoch]:
            continue
        data, target = data.cuda(), target.cuda()
        output = net(data)
        cost = criterion(output, target)
        total_loss = cost
        l1_term = l1_penalty(net)
        l1_total = l1_term * l1_param
        total_loss = total_loss + l1_total

        running_loss += total_loss.item()
        running_cost += cost.item()
        batch_loss.append(total_loss.item())
        batch_cost.append(cost.item())

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        correct_batch = (predicted == target).sum().item()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        acc_batch = 100 * correct_batch / target.size(0)
        acc_list.append(acc_batch)

    epoch_loss = running_loss / (len(train_loader) / batch_size)
    epoch_cost = running_cost / (len(train_loader) / batch_size)
    train_accuracy = 100 * correct / total

    return epoch_cost, epoch_loss, batch_cost, batch_loss, train_accuracy, acc_list


def validate(net, test_loader, epoch):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = net(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            correct_batch = (predicted == target).sum().item()
            acc_batch = 100. * correct_batch / target.size(0)
            if epoch == 1:
                val_acc_batch.append([acc_batch])
            else:
                val_acc_batch[batch_idx].append(acc_batch)

            if epoch == epochs:
                for label in range(target.shape[0]):
                    pred = predicted[label].item()
                    tg = target[label].item()
                    if pred != tg:
                        incorrect_dict[tg].append(pred)

    test_accuracy = 100 * correct / total
    if epoch % 20 == 0:
        print('Test Accuracy: {}'.format(test_accuracy))
    return test_accuracy


train_data = torch.utils.data.TensorDataset(train_samples, train_labels)
test_data = torch.utils.data.TensorDataset(test_samples, test_labels)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=min(samples_per_sbj), shuffle=False)

batches_per_epoch = int(math.ceil(train_samples.shape[0] / 50))
epochs = batches_per_epoch * 3

print(len(test_loader))
for candidates in tg_lambda:
    l1_param = candidates[0]
    l2_param = candidates[1]
    phase = 'test'
    print('++++ Starting regularization parameters {} ++++'.format(candidates))
    cand_dir = '{}/l1_{}_l2_{}'.format(output_folder, str(l1_param), str(l2_param))
    os.makedirs(cand_dir)
    os.makedirs(cand_dir + '/models')
    os.makedirs(cand_dir + '/batch_val')

    net = DNN(hidden_dim, layers_dim, output_dim)

    # net.load_state_dict(torch.load('/users/nivl/data/autoencoder/classifier/{}_fixed.pt'.format(output_dim)))

    net = net.cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum, weight_decay=l2_param)

    train_accuracies = []
    test_accuracies = []
    loss_list = []
    cost_list = []
    cost_batch = []
    loss_batch = []
    train_acc_batch = []
    lr_list = []
    val_acc_batch = []
    weight_update_dict = {}
    prev_dict = {}
    incorrect_dict = {}

    for l in range(output_dim):
        incorrect_dict[l] = []

    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        cost, loss, batch_cost, batch_loss, train_acc, batch_acc = train(net, optimizer, train_loader, l1_param)
        train_accuracies.append(train_acc)
        loss_list.append(loss)
        cost_list.append(cost)
        cost_batch.extend(batch_cost)
        loss_batch.extend(batch_loss)
        test_acc = validate(net, test_loader, epoch)
        test_accuracies.append(test_acc)
        train_acc_batch.extend(batch_acc)
        lr_list.append(optimizer.param_groups[0]['lr'])

        curr_dict = {}
        for key in net.state_dict():
            if key.endswith('weight'):
                curr_dict[key] = net.state_dict()[key].cpu().numpy()

        if epoch > 1:
            weight_update_dict = calc_weight_update(prev_dict, curr_dict, weight_update_dict)
        prev_dict = curr_dict

        if epoch % 20 == 0:
            plot_results(cost_list, loss_list, cost_batch, loss_batch, train_accuracies,
                         test_accuracies, train_acc_batch, lr_list, val_acc_batch, weight_update_dict, cand_dir)
            sio.savemat(cand_dir + '/val_acc.mat', {'accuracy': np.array(test_accuracies)})
            sio.savemat(cand_dir + '/val_acc_batch.mat', {'accuracy': np.array(val_acc_batch)})
        torch.save(net.state_dict(), cand_dir + '/models/model{}.pt'.format(epoch))

    pickle.dump(incorrect_dict, open('{}/incorrect_dict.p'.format(cand_dir), 'wb'))
    print('*** Candidate parameters: {}, Acc: {}.'.format(candidates, test_accuracies[-1]))
