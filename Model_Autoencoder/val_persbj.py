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

tasks = {'WM': 8, 'MOTOR': 5, 'EMOTION': 2, 'RELATIONAL': 2, 'SOCIAL': 2, 'LANGUAGE': 2, 'GAMBLING': 2}
task_lst = ['WM', 'MOTOR', 'EMOTION', 'RELATIONAL', 'SOCIAL', 'LANGUAGE', 'GAMBLING']
retest_sbj = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', '137128',
              '139839', '144226', '146129', '149337', '149741', '151526', '158035', '169343', '172332',
              '175439', '177746', '185442', '187547', '192439', '194140', '195041', '200109', '200614',
              '204521', '250427', '287248', '433839', '562345', '599671', '601127', '627549', '660951',
              '662551', '783462', '859671', '861456', '877168', '917255']

batch_size = 50
l2_param = 1e-4
epochs = 10
lr = 2e-3
max_b = [1e-4, 1e-4]
b_lr = [1e-5, 1e-5]
begin_anneal = 100
decay_rate = 5e-5
min_lr = 1e-4
op_type = 'SGD'
momentum = 0.3
ae_hsp = 0.9
in_dim = 52470
hidden_dim = 5000
level = 0
subjects = 100
task_id = 2
task = task_lst[task_id]
output_dim = tasks[task]
sparsity_candidates = [0.1, 0.5, 0.9]
# sparsity_candidates = [0.9]
tg_hspset_list = list(itertools.product(sparsity_candidates, sparsity_candidates))
tg_hspset_list = [list(i) for i in tg_hspset_list]
tg_hsp = [0.2, 0.2]
layers_dim = [1000, 500]
load_model = False
pretrained_path = '/users/nivl/data/autoencoder/classifier/0622'
sp_layers = 2

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
output_folder = '/users/nivl/data/autoencoder/classifier/examine_val/{}{}/{}{}{}_{}_hsp{}_pct{}'.format(month, day, hour, minute, sec, task, hsp_str, pct_str)
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
f.write('L2_reg : ' + str(l2_param) + '\n')
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
random.shuffle(hcp_sbj)
enc = Encoder(in_dim, hidden_dim).cuda()

pretrained_dict = torch.load('/users/nivl/data/autoencoder/hsp/all/hsp{}_{}pct/models/epoch_10.pt'.format(hsp_str, pct_str))
enc_dict = enc.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in enc_dict}
# 2. overwrite entries in the existing state dict
enc_dict.update(pretrained_dict)
# 3. load the new state dict
enc.load_state_dict(enc_dict)

for param in enc.parameters():
    param.requires_grad = False

train_samples = None
train_labels = None

test_samples = []
test_labels = []

sbj_idx = 0
sbj_loaded = 0

for sbj in hcp_sbj:
    # if sbj_loaded > 600:
    #     break
    if not is_subject_valid(sbj, task, False):
        continue
    sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)
    sbj_samples = torch.from_numpy(sbj_samples).cuda()
    enc_output = enc(sbj_samples).cpu()
    sbj_samples = sbj_samples.cpu()

    if sbj_loaded > 199:
        if train_samples is None:
            train_samples = enc_output
            train_labels = sbj_labels
        else:
            train_samples = torch.cat((train_samples, enc_output))
            train_labels = np.concatenate((train_labels, sbj_labels))
    else:
        test_samples.append(enc_output)
        test_labels.append(torch.from_numpy(sbj_labels))
    sbj_loaded += 1
    print(sbj_loaded)

print('loaded data')

training_labels = torch.from_numpy(train_labels)

train_data = torch.utils.data.TensorDataset(train_samples, training_labels)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


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


def get_pretrained_path(path, mode):
    runs = os.listdir(path)
    for run in runs:
        if all([substring in run for substring in [task, 'hsp' + hsp_str, 'pct' + pct_str]]):
            run_path = '{}/{}'.format(path, run)
            model_path = '{}/{}/model100.pt'.format(run_path, mode)
            return model_path


def plot_results(cost_list, loss_list, batchcost, batchloss, sparsity, betas, batch_accuracies, out_dir):
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


def plot_validation_accuracies(epoch_dict, batch_dict, out_dir):
    batch_dir = out_dir + '/batch_acc'
    epoch_dir = out_dir + '/epoch_acc'
    if not os.path.isdir(epoch_dir):
        os.makedirs(epoch_dir)
        os.makedirs(batch_dir)

    for i in range(len(epoch_dict)):
        plot = plt.figure()
        acc_batch = batch_dict[i]
        plt.title('Validation accuracy per batch subject {0}: {1:.3f}'.format(i, batch_dict[i][-1]))
        plt.plot(range(len(acc_batch)), acc_batch, c='orange')
        plot.savefig('{}/{}.png'.format(batch_dir, i))
        plt.close()


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


def l1_penalty(model, hsp_val, beta_val, max_b, b_lr, tg_hsp):
    model_layers = [x for x in list(model.parameters()) if len(x.shape) == 2]
    l1_reg = None
    layer_idx = 0
    for layer in range(2):
        W = model_layers[layer]
        hsp_val[layer_idx], beta_val[layer_idx] = hsp_sparsity_control(W, beta_val[layer_idx], max_b[layer], b_lr[layer], tg_hsp[layer])
        layer_l1 = torch.sum(torch.abs(torch.t(W)) * beta_val[layer_idx])
        if l1_reg is None:
            l1_reg = layer_l1
        else:
            l1_reg = l1_reg + layer_l1
        layer_idx += 1

    return l1_reg, hsp_val, beta_val


def validate(model, test_samples, test_labels, phase):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for test_sbj_idx in range(len(test_samples)):
        data, target = test_samples[test_sbj_idx], test_labels[test_sbj_idx]
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            correct_sbj = (predicted == target).sum().item()
            sbj_acc = 100 * correct_sbj / target.size(0)
            if phase == 'epoch':
                epoch_acc_dict[test_sbj_idx].append(sbj_acc)
            else:
                batch_acc_dict[test_sbj_idx].append(sbj_acc)

    test_accuracy = 100 * correct / total
    if phase == 'epoch':
        print('Test Accuracy: {}'.format(test_accuracy))
    return test_accuracy


def train(model, optimizer, train_loader, beta_val, tg_hsp):
    model.train()
    criterion = nn.NLLLoss()
    running_loss = 0
    running_cost = 0
    correct = 0
    total = 0
    hsp_val = [None, None]
    batch_cost = []
    batch_loss = []
    acc_list = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        cost = criterion(output, target)
        total_loss = cost
        l1_term, hsp_val, beta_val = l1_penalty(model, hsp_val, beta_val, max_b, b_lr, tg_hsp)

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
        _ = validate(model, test_samples, test_labels, 'batch')
        acc_batch = 100 * correct_batch / target.size(0)
        acc_list.append(acc_batch)

        if epoch < 3:
            if batch_idx == 9 or batch_idx % 50 == 0:
                torch.save(model.state_dict(), batch_models_dir + '/model{}_{}.pt'.format(epoch, batch_idx))
    epoch_loss = running_loss / (len(train_loader) / batch_size)
    epoch_cost = running_cost / (len(train_loader) / batch_size)
    train_accuracy = 100 * correct / total
    meanhsp = [torch.mean(hsp_val[j]).item() for j in range(len(hsp_val))]
    print("======> epoch: {}/{}. Avg. HSP: {}, Loss: {}, Cost: {}, Train Acc: {}".format(epoch, epochs, meanhsp, epoch_loss, epoch_cost, train_accuracy))

    return epoch_cost, epoch_loss, batch_cost, batch_loss, train_accuracy, hsp_val, beta_val, acc_list


tghsp = ''.join(i for i in str(tg_hsp) if i.isdigit())
hsp_dir = '{}/{}'.format(output_folder, tghsp)
batch_models_dir = '{}/batch_models'.format(hsp_dir)

os.makedirs(hsp_dir)
os.makedirs(batch_models_dir)

epoch_acc_dict = []
batch_acc_dict = []

for m in range(200):
    epoch_acc_dict.append([])
    batch_acc_dict.append([])

net = DNN(hidden_dim, layers_dim, output_dim)

net.load_state_dict(torch.load('/users/nivl/data/autoencoder/classifier/{}_fixed.pt'.format(output_dim)))

net = net.cuda()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum, weight_decay=l2_param)

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

for l in range(sp_layers):
    sparsity.append(np.zeros((epochs, layers_dim[l])))
    betas.append(np.zeros((epochs, layers_dim[l])))

for epoch in range(1, epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    cost, loss, batch_cost, batch_loss, train_acc, hsp_val, beta_val, batch_acc = train(net, optimizer, train_loader, beta_val, tg_hsp)
    train_accuracies.append(train_acc)
    loss_list.append(loss)
    cost_list.append(cost)
    cost_batch.extend(batch_cost)
    loss_batch.extend(batch_loss)
    test_acc = validate(net, test_samples, test_labels, 'epoch')
    test_accuracies.append(test_acc)
    train_acc_batch.extend(batch_acc)

    for idx in range(sp_layers):
        sparsity[idx][epoch - 1] = torch.clone(hsp_val[idx][0]).detach().cpu().numpy()
        betas[idx][epoch - 1] = torch.clone(beta_val[idx][0]).detach().cpu().numpy()

    plot_results(cost_list, loss_list, cost_batch, loss_batch, sparsity, betas, train_acc_batch, hsp_dir)
    torch.save(net.state_dict(), hsp_dir + '/model{}.pt'.format(epoch))

    plot_validation_accuracies(epoch_acc_dict, batch_acc_dict, hsp_dir)
