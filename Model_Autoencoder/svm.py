from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from thundersvm import SVC
import os
import random
from data_utils import get_sbj_task_data, is_subject_valid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# print(__doc__)
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


task_id = 6
task = task_lst[task_id]
reduce_dim = True

ae_hsp = 0.8
level = 0
hsp_str = str(ae_hsp)
hsp_str = hsp_str.replace('.', '')
pct_str = str(int(level * 100))

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
    pretrained_dict = torch.load(
        '/users/nivl/data/autoencoder/hsp/all/hsp{}_{}pct/models/epoch_10.pt'.format(hsp_str, pct_str))
    enc_dict = enc.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in enc_dict}
    # 2. overwrite entries in the existing state dict
    enc_dict.update(pretrained_dict)
    # 3. load the new state dict
    enc.load_state_dict(enc_dict)

for sbj in hcp_sbj:
    # if sbj_loaded >= 200:
    #     break
    if not is_subject_valid(sbj, task, False):
        continue
    sbj_samples, sbj_labels = get_sbj_task_data(sbj, task)
    if len(sbj_labels) not in [271]:
        continue
    # if len(sbj_labels) == 224:
    #     sbj_samples = sbj_samples[0:-1]
    #     sbj_labels = sbj_labels[0:-1]

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
parameters_list = [[{'kernel': ['rbf'], 'gamma': [2**x for x in range(-5, 16, 2)], 'C': [2**x for x in range(-15, 4, 2)]}],
                    [{'kernel': ['linear'], 'C': [2**x for x in range(-15, 4, 2)]}]]
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**x for x in range(-5, 16, 2)], 'C': [2**x for x in range(-15, 4, 2)]}]
# tuned_parameters = [{'kernel': ['linear'], 'C': [2**x for x in range(-15, 4, 2)]}]

print('Input dimension is {}'.format(train_samples.shape[1]))
print('===== Task: {} ====='.format(task_lst[task_id]))
for tuned_parameters in parameters_list:
    clf = GridSearchCV(SVC(), tuned_parameters, verbose=100, cv=4, scoring='accuracy')
    clf.fit(train_samples, train_labels)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()

    predict = clf.predict(test_samples)

    count_correct = 0
    for index in range(0, len(test_labels)):
        if predict[index] == test_labels[index]:
            count_correct = count_correct + 1
    accuracy = count_correct / len(test_labels) * 100
    print('Final test Accuracy: {}'.format(accuracy))
    # y_true, y_pred = test_labels, clf.predict(test_samples)
    # print(classification_report(y_true, y_pred))
    # print()
