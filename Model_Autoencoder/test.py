import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib
import scipy.io as sio

path = '/users/nivl/data/autoencoder/classifier/fixed/1103'


for run in os.listdir(path):
    best_cand = ''
    best_acc = 0
    best_std = 0

    run_dir = '{}/{}'.format(path, run)
    for candidates in os.listdir(run_dir):
        if not (candidates.startswith('l1') or candidates.startswith('hsp')):
            continue
        cand_dir = '{}/{}'.format(run_dir, candidates)
        acc_f = cand_dir + '/val_acc.mat'
        if not os.path.isfile(acc_f):
            continue
        mat = sio.loadmat(cand_dir + '/val_acc_batch.mat')
        batch_accuracies = mat['accuracy']
        acc = np.mean(batch_accuracies[:, -1])
        if acc > best_acc:
            best_cand = candidates
            best_acc = acc
            best_std = np.std(batch_accuracies[:, -1])

    print(run)
    print(best_acc)
    print(best_cand)
    print(best_std)
    print('\n')

# path = '/users/nivl/data/autoencoder/classifier/finetune/0917/191722_hsp08_pct0_rbm'
#
# for run in os.listdir(path):
#     best_acc = 0
#     best_std = 0
#     if run == 'parameters.txt':
#         continue
#
#     run_dir = '{}/{}'.format(path, run)
#
#     acc_file = run_dir + '/val_acc_batch.mat'
#     if not os.path.isfile(acc_file):
#         continue
#     mat = sio.loadmat(run_dir + '/val_acc_batch.mat')
#     batch_accuracies = mat['accuracy']
#     acc = np.mean(batch_accuracies[:, -1])
#     if acc > best_acc:
#         best_acc = acc
#         best_std = np.std(batch_accuracies[:, -1])
#
#     print(run)
#     print(best_acc)
#     print(best_std)
#     print('\n')



