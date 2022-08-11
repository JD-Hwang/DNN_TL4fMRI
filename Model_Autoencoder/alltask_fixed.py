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
from data_utils import get_sbj_data_alltask, is_subject_valid
import timeit
import itertools
import collections
import scipy.io as sio
from numpy import linalg as LA
import pickle


retest_sbj = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', '137128',
              '139839', '144226', '146129', '149337', '149741', '151526', '158035', '169343', '172332',
              '175439', '177746', '185442', '187547', '192439', '194140', '195041', '200109', '200614',
              '204521', '250427', '287248', '433839', '562345', '599671', '601127', '627549', '660951',
              '662551', '783462', '859671', '861456', '877168', '917255']

batch_size = 50
epochs = 100
lr = 5e-4
begin_anneal = 25
decay_rate = 2e-3
min_lr = 5e-5
op_type = 'SGD'
momentum = 0.1
ae_hsp = 0.8
in_dim = 52470
hidden_dim = 5000
level = 0
l2_param = 1e-5
hsp_list = [0.8, 0.5, 0.9]
max_b = 1e-3
b_lr = 1e-4
sp_layers = 1
layers_dim = [1000]
output_dim = 23
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

output_folder = '/users/nivl/data/autoencoder/classifier/fixed/alltasks/{}{}/{}{}{}_hsp{}_pct{}_{}_{}labels'.format(month, day, hour, minute, sec, hsp_str, pct_str, model, output_dim)
# output_folder = '/users/nivl/data/dnn_results/DEBUG/{}{}{}_{}{}{}'.format(str(dt.now().year), month, day, hour, minute, sec)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

f = open(output_folder + "/parameters.txt", 'w')
f.write('optimizer_algorithm : ' + str(op_type) + '\n')
f.write('n_epochs : ' + str(epochs) + '\n')
f.write('batch_size : ' + str(batch_size) + '\n')
f.write('init_learning_rate : '+str(lr)+'\n')
f.write('begin_anneal : ' + str(begin_anneal) + '\n')
f.write('decay_rate : ' + str(decay_rate) + '\n')
f.write('min_lr : ' + str(min_lr) + '\n')
f.write('HSP targers : ' + str(hsp_list) + '\n')
f.write('momentum : ' + str(momentum) + '\n')
f.write('beta lr : ' + str(b_lr) + '\n')
f.write('max_beta : ' + str(max_b) + '\n')
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
    # if sbj_loaded > 10:
    #     break
    if not is_subject_valid(sbj, 'all', False):
        continue
    sbj_samples, sbj_labels = get_sbj_data_alltask(sbj)

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

test_samples = []
test_labels = []
sbj_loaded = 0
for sbj in retest_sbj:
    # if sbj_loaded > 4:
    #     continue
    if not is_subject_valid(sbj, 'all', True):
        continue
    sbj_samples, sbj_labels = get_sbj_data_alltask(sbj)
    sbj_samples = torch.from_numpy(sbj_samples).cuda()
    enc_output = enc(sbj_samples).cpu()
    sbj_samples = sbj_samples.cpu()
    test_samples.append(enc_output)
    test_labels.append(sbj_labels)
    sbj_loaded += 1

print("loaded test data")

train_labels = torch.from_numpy(train_labels)


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


def plot_results(cost_list, loss_list, batchcost, batchloss, train_accuracies, test_accuracies, batch_accuracies, lr_list, batch_val, weight_dict, sp_lst, beta_lst, out_dir):
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

    for l in range(len(sp_lst)):
        plot = plt.figure()
        plt.title('HSP value')
        for i in range(sp_lst[l].shape[1]):
            node_hsp_vec = sp_lst[l][:epoch, i]
            plt.plot(node_hsp_vec)
        plot.savefig('{}/hsp_{}.png'.format(out_dir, l))
        plt.close()

        plot = plt.figure()
        plt.title('Beta value')
        for i in range(betas[l].shape[1]):
            node_beta_vec = beta_lst[l][:epoch, i]
            plt.plot(node_beta_vec)
        plot.savefig('{}/beta_{}.png'.format(out_dir, l))
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


def l1_penalty(model, hsp_val, tg_hsp, beta_val):
    model_layers = [x for x in list(model.parameters()) if len(x.shape) == 2]

    W = model_layers[0]
    hsp_val[0], beta_val[0] = hsp_sparsity_control(W, beta_val[0], max_b, b_lr, tg_hsp)
    layer_l1 = torch.sum(torch.abs(torch.t(W)) * beta_val[0])

    return layer_l1, hsp_val, beta_val


def train(net, optimizer, train_loader, tg_hsp, beta_val):
    net.train()
    criterion = nn.NLLLoss()
    running_loss = 0
    running_cost = 0
    correct = 0
    total = 0
    batch_cost = []
    batch_loss = []
    acc_list = []
    hsp_val = [None]
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = net(data)
        cost = criterion(output, target)
        total_loss = cost

        l1_term, hsp_val, beta_val = l1_penalty(net, hsp_val, tg_hsp, beta_val)

        total_loss = total_loss + l1_term

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

    return epoch_cost, epoch_loss, batch_cost, batch_loss, train_accuracy, hsp_val, beta_val, acc_list


def validate(net, samples, labels, epoch):
    net.eval()
    correct = 0
    total = 0
    for batch_idx in range(len(samples)):
        data, target = samples[batch_idx], torch.from_numpy(labels[batch_idx])
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
    return test_accuracy


train_data = torch.utils.data.TensorDataset(train_samples, train_labels)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

for tg_hsp in hsp_list:
    phase = 'test'
    print('++++ Starting sparsity target parameter {} ++++'.format(tg_hsp))
    cand_str = str(tg_hsp).replace('.', '')
    cand_dir = '{}/hsp_{}'.format(output_folder, cand_str)
    os.makedirs(cand_dir)
    # os.makedirs(cand_dir + '/models')
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
    beta_val = [None]
    sparsity = []
    betas = []

    for l in range(output_dim):
        incorrect_dict[l] = []

    for l in range(sp_layers):
        sparsity.append(np.zeros((epochs, layers_dim[l])))
        betas.append(np.zeros((epochs, layers_dim[l])))

    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        cost, loss, batch_cost, batch_loss, train_acc, hsp_val, beta_val, batch_acc = train(net, optimizer, train_loader, tg_hsp, beta_val)
        train_accuracies.append(train_acc)
        loss_list.append(loss)
        cost_list.append(cost)
        cost_batch.extend(batch_cost)
        loss_batch.extend(batch_loss)
        test_acc = validate(net, test_samples, test_labels, epoch)
        test_accuracies.append(test_acc)
        train_acc_batch.extend(batch_acc)
        lr_list.append(optimizer.param_groups[0]['lr'])

        curr_dict = {}
        for key in net.state_dict():
            if key.endswith('weight'):
                curr_dict[key] = net.state_dict()[key].cpu().numpy()

        for idx in range(sp_layers):
            sparsity[idx][epoch - 1] = torch.clone(hsp_val[idx]).detach().cpu().numpy()
            betas[idx][epoch - 1] = torch.clone(beta_val[idx]).detach().cpu().numpy()

        mean_hsp = [torch.mean(hsp_val[j]).item() for j in range(len(hsp_val))]
        if epoch > 1:
            weight_update_dict = calc_weight_update(prev_dict, curr_dict, weight_update_dict)
        prev_dict = curr_dict

        if epoch % 20 == 0:
            plot_results(cost_list, loss_list, cost_batch, loss_batch, train_accuracies,
                         test_accuracies, train_acc_batch, lr_list, val_acc_batch, weight_update_dict, sparsity, betas, cand_dir)
            print('*** tg sparsity: {}, Acc: {}.'.format(mean_hsp, test_accuracies[-1]))
            torch.save(net.state_dict(), cand_dir + '/model.pt')

    sio.savemat(cand_dir + '/val_acc.mat', {'accuracy': np.array(test_accuracies)})
    sio.savemat(cand_dir + '/val_acc_batch.mat', {'accuracy': np.array(val_acc_batch)})
    pickle.dump(incorrect_dict, open('{}/incorrect_dict.p'.format(cand_dir), 'wb'))
    print('*** Final tg sparsity: {}, Acc: {}.'.format(mean_hsp, test_accuracies[-1]))
