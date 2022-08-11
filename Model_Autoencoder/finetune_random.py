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
from data_utils import get_sbj_task_data, is_subject_valid
import collections
import scipy.io as sio
import pickle
import scipy.stats as stats

tasks = {'WM': 8, 'MOTOR': 5, 'EMOTION': 2, 'RELATIONAL': 2, 'SOCIAL': 2, 'LANGUAGE': 2, 'GAMBLING': 2}
task_lst = ['WM', 'MOTOR', 'EMOTION', 'RELATIONAL', 'SOCIAL', 'LANGUAGE', 'GAMBLING']

retest_sbj = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', '137128',
              '139839', '144226', '146129', '149337', '149741', '151526', '158035', '169343', '172332',
              '175439', '177746', '185442', '187547', '192439', '194140', '195041', '200109', '200614',
              '204521', '250427', '287248', '433839', '562345', '599671', '601127', '627549', '660951',
              '662551', '783462', '859671', '861456', '877168', '917255']

batch_size = 50
epochs = 100
lr = 1e-3
max_b = 7e-5
b_lr = 7e-6
begin_anneal = 25
decay_rate = 5e-3
min_lr = 1e-4
op_type = 'SGD'
momentum = 0.3
in_dim = 52470
hidden_dim = 5000
level = 0
ae_hsp = 0.8

layers_dim = [5000, 1000]
sp_layers = 1
mode = 'finetune'
pretrained_model = 'ae'


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
output_folder = '/users/nivl/data/autoencoder/classifier/{}/{}{}/{}{}{}_hsp{}_pct{}_{}'.format(mode, month, day, hour, minute, sec, hsp_str, pct_str, pretrained_model)
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
f.write('beta_lrate : ' + str(b_lr) + '\n')
f.write('max_beta : ' + str(max_b) + '\n')
f.close()


class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.h1 = nn.Linear(hidden_dim, 1000)
        self.out = nn.Linear(1000, output_dim)
        self._initialize_parameters()

    def forward(self, x):
        x = F.relu(self.encoder(x), inplace=True)
        x = F.relu(self.h1(x), inplace=True)
        x = self.out(x)
        x = F.log_softmax(x, dim=1)
        return x

    def make_layers(self, hidden_layers, num_classes, input_dim):
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


hcp_sbj = [e for e in os.listdir('/data4/open_data/HCP') if e.isdecimal()]
random.shuffle(hcp_sbj)


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


def plot_results(cost_list, loss_list, batchcost, batchloss, sparsity, betas, train_accuracies, test_accuracies, batch_accuracies, diff_dict, out_dir):
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
    plt.ylim(0, 100)
    plot.savefig('{}/batch_accuracy.png'.format(out_dir))
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

    for l in range(len(sparsity)):
        plot = plt.figure()
        plt.title('HSP value')
        for i in range(sparsity[l].shape[1]):
            node_hsp_vec = sparsity[l][:epoch, i]
            plt.plot(node_hsp_vec)
        plot.savefig('{}/hsp_{}.png'.format(out_dir, l))
        plt.close()

        plot = plt.figure()
        plt.title('Beta value')
        for i in range(betas[l].shape[1]):
            node_beta_vec = betas[l][:epoch, i]
            plt.plot(node_beta_vec)
        plot.savefig('{}/beta_{}.png'.format(out_dir, l))
        plt.close()

    for key, value in diff_dict.items():
        filename = key.replace('.', '_')
        plot = plt.figure()
        plt.plot(value[1:])
        plot.savefig('{}/{}.png'.format(out_dir, filename))
        plt.close()


def get_weight_diff(model, prev):
    diff_batch = {}
    model_dict = model.state_dict()
    for key in t.keys():
        if 'weight' in key:
            curr_w = model_dict[key].detach().clone().cpu()
            prev_w = prev_dict[key]
            update_w = torch.norm((curr_w - prev_w), p=2) / torch.norm(curr_w, p=2)
            diff_batch[key] = update_w.item()
            prev_dict[key] = curr_w

    return diff_batch, prev


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


def l1_penalty(model, hsp_val, beta_val):
    model_layers = [x for x in list(model.parameters()) if len(x.shape) == 2]

    W = model_layers[0]
    hsp_val[0], beta_val[0] = hsp_sparsity_control(W, beta_val[0], max_b, b_lr, ae_hsp)
    layer_l1 = torch.sum(torch.abs(torch.t(W)) * beta_val[0])

    return layer_l1, hsp_val, beta_val


def train(model, optimizer, train_loader, beta_val):
    model.train()
    criterion = nn.NLLLoss()
    running_loss = 0
    running_cost = 0
    correct = 0
    total = 0
    hsp_val = [None] * sp_layers
    batch_cost = []
    batch_loss = []
    acc_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        cost = criterion(output, target)
        total_loss = cost
        l1_term, hsp_val, beta_val = l1_penalty(model, hsp_val, beta_val)

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
    meanhsp = [torch.mean(hsp_val[j]).item() for j in range(len(hsp_val))]

    print("======> epoch: {}/{}. Avg. HSP: {}, Loss: {}, Cost: {}, Train Acc: {}".format(epoch, epochs, meanhsp, epoch_loss, epoch_cost, train_accuracy))

    return epoch_cost, epoch_loss, batch_cost, batch_loss, train_accuracy, hsp_val, beta_val, acc_list


def validate(model, samples, labels, epoch):
    model.eval()
    correct = 0
    total = 0
    for batch_idx in range(len(samples)):
        data, target = samples[batch_idx], labels[batch_idx]
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct_batch = (predicted == target).sum().item()
            correct += correct_batch
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

    test_accuracy = 100. * correct / total
    if epoch % 10 == 0:
        print('Test Accuracy: {}'.format(test_accuracy))
    return test_accuracy


for task_id in range(7):
    task = task_lst[task_id]
    # if task_id < 1:
    #     continue
    print('++++ Starting {} task++++'.format(task))
    train_samples = None
    train_labels = None

    test_samples = []
    test_labels = []

    sbj_idx = 0
    sbj_loaded = 0

    for sbj in hcp_sbj:
        if sbj_loaded > 50:
            break
        if not is_subject_valid(sbj, task, False):
            continue

        sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)

        if train_samples is None:
            train_samples = sbj_samples
            train_labels = sbj_labels
        else:
            train_samples = np.concatenate((train_samples, sbj_samples))
            train_labels = np.concatenate((train_labels, sbj_labels))
        sbj_loaded += 1

    train_samples = torch.from_numpy(train_samples)
    train_labels = torch.from_numpy(train_labels)
    print('Loaded {} training subjects'.format(sbj_loaded))

    sbj_loaded = 0
    for sbj in retest_sbj:
        if sbj_loaded > 4:
            break
        if not is_subject_valid(sbj, task, True):
            continue
        sbj_test_samples, sbj_test_labels = get_sbj_task_data(sbj, task)
        sbj_retest_samples, sbj_retest_labels = get_sbj_task_data(sbj, task, retest=True)
        sbj_samples = np.concatenate((sbj_test_samples, sbj_retest_samples))
        sbj_labels = np.concatenate((sbj_test_labels, sbj_retest_labels))

        test_samples.append(torch.from_numpy(sbj_samples))
        test_labels.append(torch.from_numpy(sbj_labels))
        sbj_loaded += 1

    print('Loaded {} test subjects'.format(sbj_loaded))

    task_dir = '{}/{}'.format(output_folder, str(task))
    os.makedirs(task_dir)

    train_data = torch.utils.data.TensorDataset(train_samples, train_labels)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    output_dim = tasks[task]
    net = DNN(in_dim, hidden_dim, output_dim)

    # net.load_state_dict(torch.load('/users/nivl/data/autoencoder/classifier/{}_full.pt'.format(output_dim)))

    if mode == 'finetune':
        if pretrained_model == 'ae':
            pretrain_path = '/users/nivl/data/autoencoder/hsp/all/hsp{}_{}pct/models/epoch_10.pt'.format(hsp_str,
                                                                                                         pct_str)
            pretrained_dict = torch.load(pretrain_path)
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            # 2. overwrite entries in the existing state dict
            net_dict.update(pretrained_dict)
            # 3. load the new state dict
            net.load_state_dict(net_dict)
        else:
            mat = sio.loadmat('/users/nivl/data/autoencoder/dbn/epoch15/1_15epoc.mat')
            weight = torch.from_numpy(mat['tmp_w'])
            bias = torch.flatten(torch.from_numpy(mat['tmp_c']))
            state_dict = net.state_dict()
            state_dict['encoder.weight'] = weight
            state_dict['encoder.bias'] = bias
            net.load_state_dict(state_dict)

    net = net.cuda()
    prev_dict = {}
    diff_dict = {}
    t = net.state_dict()
    for key in t.keys():
        if 'weight' in key:
            prev_dict[key] = torch.zeros(t[key].shape)
            diff_dict[key] = []

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum, weight_decay=1e-5)

    train_accuracies = []
    test_accuracies = []
    loss_list = []
    cost_list = []
    cost_batch = []
    loss_batch = []
    beta_val = [None for i in range(sp_layers)]
    sparsity = []
    betas = []
    train_acc_batch = []
    val_acc_batch = []

    for l in range(sp_layers):
        sparsity.append(np.zeros((epochs, layers_dim[l])))
        betas.append(np.zeros((epochs, layers_dim[l])))

    incorrect_dict = {}

    for l in range(output_dim):
        incorrect_dict[l] = []

    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        cost, loss, batch_cost, batch_loss, train_acc, hsp_val, beta_val, batch_acc = train(net, optimizer, train_loader, beta_val)
        weight_diff, prev_dict = get_weight_diff(net, prev_dict)
        for key in net.state_dict().keys():
            if 'weight' in key:
                diff_dict[key].append(weight_diff[key])

        train_accuracies.append(train_acc)
        loss_list.append(loss)
        cost_list.append(cost)
        cost_batch.extend(batch_cost)
        loss_batch.extend(batch_loss)
        test_acc = validate(net, test_samples, test_labels, epoch)
        test_accuracies.append(test_acc)
        train_acc_batch.extend(batch_acc)

        for idx in range(sp_layers):
            sparsity[idx][epoch - 1] = torch.clone(hsp_val[idx]).detach().cpu().numpy()
            betas[idx][epoch - 1] = torch.clone(beta_val[idx]).detach().cpu().numpy()

        if epoch % 10 == 0:
            plot_results(cost_list, loss_list, cost_batch, loss_batch, sparsity, betas, train_accuracies, test_accuracies, train_acc_batch, diff_dict, task_dir)

    mean_hsp = [torch.mean(hsp_val[j]).item() for j in range(len(hsp_val))]

    torch.save(net.state_dict(), task_dir + '/model.pt')
    sio.savemat(task_dir + '/val_acc.mat', {'accuracy': np.array(test_accuracies)})
    sio.savemat(task_dir + '/val_acc_batch.mat', {'accuracy': np.array(val_acc_batch)})
    pickle.dump(incorrect_dict, open('{}/incorrect_dict.p'.format(task_dir), 'wb'))
    print('*** Target sparsity: {}, Acc: {}. HSP: {}'.format(ae_hsp, test_accuracies[-1], mean_hsp))
