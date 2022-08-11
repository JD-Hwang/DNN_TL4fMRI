import os
import random
from data_utils import get_sbj_task_data, is_subject_valid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import time
# from thundersvm import SVC
from sklearn.svm import SVC


tasks = {'WM': 8, 'MOTOR': 5, 'EMOTION': 2, 'RELATIONAL': 2, 'SOCIAL': 2, 'LANGUAGE': 2, 'GAMBLING': 2}
task_lst = ['WM', 'MOTOR', 'EMOTION', 'RELATIONAL', 'SOCIAL', 'LANGUAGE', 'GAMBLING']
retest_sbj = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', '137128',
              '139839', '144226', '146129', '149337', '149741', '151526', '158035', '169343', '172332',
              '175439', '177746', '185442', '187547', '192439', '194140', '195041', '200109', '200614',
              '204521', '250427', '287248', '433839', '562345', '599671', '601127', '627549', '660951',
              '662551', '783462', '859671', '861456', '877168', '917255']


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.encoder(x), inplace=True)
        return x


n_folds = 4
task_id = 2
task = task_lst[task_id]
reduce_dim = True
kernel = 'linear'
C_list = [2**x for x in range(-5, 16, 2)]
gamma_list = [2**x for x in range(-15, 4, 2)]
candidate_list = list(itertools.product(C_list, gamma_list)) if kernel == 'rbf' else C_list
hcp_sbj = [e for e in os.listdir('/data4/open_data/HCP') if e.isdecimal()]
# random.shuffle(hcp_sbj)

train_samples = None
train_labels = None

test_samples = None
test_labels = None

sbj_idx = 0
sbj_loaded = 0

if reduce_dim:
    enc = Encoder(52470, 5000).cuda()

for sbj in hcp_sbj:
    # if sbj_loaded >= 200:
    #     break
    if not is_subject_valid(sbj, task, False):
        continue
    sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)
    if len(sbj_labels) not in [223, 224]:
        continue
    if len(sbj_labels) == 224:
        sbj_samples = sbj_samples[0:-1]
        sbj_labels = sbj_labels[0:-1]

    if reduce_dim:
        sbj_samples = torch.from_numpy(sbj_samples).cuda()
        enc_output = enc(sbj_samples).cpu()
        sbj_samples = enc_output.cpu().detach().numpy()

    if train_samples is None:
        train_samples = sbj_samples
        train_labels = sbj_labels
    else:
        train_samples = np.concatenate((train_samples, sbj_samples))
        train_labels = np.concatenate((train_labels, sbj_labels))
    sbj_loaded += 1
    print(sbj_loaded)

print('loaded training data')

sbj_loaded = 0
for sbj in retest_sbj:
    # if sbj_loaded >= 5:
    #     break
    if not is_subject_valid(sbj, task, True):
        continue
    sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)

    if reduce_dim:
        sbj_samples = torch.from_numpy(sbj_samples).cuda()
        enc_output = enc(sbj_samples).cpu()
        sbj_samples = enc_output.cpu().detach().numpy()

    if test_samples is None:
        test_samples = sbj_samples
        test_labels = sbj_labels
    else:
        test_samples = np.concatenate((test_samples, sbj_samples))
        test_labels = np.concatenate((test_labels, sbj_labels))
    sbj_loaded += 1
    print(sbj_loaded)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5], 'C': [0.01, 0.1, 1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]


print('Input dimension is {}'.format(train_samples.shape[1]))
print('Kernel used: {}'.format(kernel))
split = np.array_split(range(len(train_labels)), 4)

best_acc = 0
best_cand = None
for candidate in candidate_list:
    print('==== Beginning candidate {} ===='.format(candidate))
    folds_acc = []
    for fold in range(n_folds):
        t0 = time.time()
        train_folds = [0, 1, 2, 3]
        train_folds.remove(fold)
        val_ids = split[fold]
        train_ids = list(split[train_folds[0]]) + list(split[train_folds[1]]) + list(split[train_folds[2]])

        val_samples = train_samples[val_ids]
        val_labels = train_labels[val_ids]

        inner_train_samples = train_samples[train_ids]
        inner_train_labels = train_labels[train_ids]

        if kernel == 'linear':
            clf_val = SVC(C=candidate, kernel=kernel)
        else:
            clf_val = SVC(C=candidate[0], kernel=kernel, gamma=candidate[1])
        train_error = clf_val.fit(inner_train_samples, inner_train_labels)
        predict = clf_val.predict(val_samples)

        count_correct = 0
        for index in range(0, len(val_labels)):
            if predict[index] == val_labels[index]:
                count_correct = count_correct + 1
        accuracy_fold = count_correct / len(val_labels) * 100

        t1 = time.time()
        total = (t1 - t0) / 60
        print('--> Fold {} validation accuracy: {}. Fold time: {} minutes'.format(fold + 1, accuracy_fold, total))
        folds_acc.append(accuracy_fold)
    cand_acc = np.mean(folds_acc)
    if cand_acc > best_acc:
        best_acc = cand_acc
        best_cand = candidate

print("Best parameters set found on validation set:")
print()
print(best_cand)
print()
print("Best score on validation set:")
print(best_acc)
print()

print("The model is trained on the full training set.")
print("The scores are computed on the full test set.")
print()

if kernel == 'linear':
    clf_test = SVC(C=best_cand, kernel=kernel)
else:
    clf_test = SVC(C=best_cand[0], kernel=kernel, gamma=best_cand[1])

predict = clf_test.predict(test_samples)

count_correct = 0
for index in range(0, len(test_labels)):
    if predict[index] == test_labels[index]:
        count_correct = count_correct + 1
accuracy = count_correct / len(test_labels) * 100
print('Final test Accuracy: {}'.format(accuracy))
