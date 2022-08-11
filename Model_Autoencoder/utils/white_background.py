import numpy
import torch
import numpy as np
import scipy.io as sio
import os
import scipy.stats as stats
import random
import pickle
import math
import torch.nn as nn
import collections
import matplotlib.pyplot as plt
import torch.nn.functional as F
import nibabel as nib
import shutil
# from dbn.extract_7nets import get_assigned_mat

# fl = np.load('/users/nivl/code/DAE/data/flip_yeo7.npz')
# f_nets = fl['yeo7']


def get_assigned_mat(path, method='zval'):
    img = nib.load('/users/nivl/code/DAE/data/flipped_hcp_8xx.nii')
    mask = img.get_fdata()
    idx = np.where(mask.flatten() == 1)[0]
    model = torch.load(path)
    layer_list = []

    for key, value in model.items():
        if 'weight' in key:
            layer_list.append(value.cpu().numpy())

    tmp = layer_list[0]

    zscored = stats.zscore(tmp, axis=None, ddof=1)

    thrshed = zscored.copy()
    if method == 'zval':
        i, j = np.where(thrshed < 1.96)
        thrshed[i, j] = 0
    else:
        pct_val = np.percentile(thrshed, 95)
        thrshed[thrshed < pct_val] = 0

    # max assign
    vec_assigned = np.zeros((52470))
    for i, position in enumerate(thrshed.transpose()):
        if np.sum(position) == 0:
            continue
        n = np.argmax(position)
        vec_assigned[i] = n + 1

    # mapping
    vol = np.zeros((61 * 73 * 61))
    vol[idx] = vec_assigned

    assignmax = np.reshape(vol, [61, 73, 61])
    return assignmax


fl = np.load('/users/nivl/code/DAE/data/mask_fnets_intersect.npz')
f_nets = fl['f_nets']
# mask = fl['mask']
img = nib.load('/users/nivl/code/DAE/data/flipped_hcp_8xx.nii')
mask = img.get_fdata()
# f_nets[mask != 0] = 0
base_path = '/users/nivl/data/autoencoder/classifier/finetune/0907/075346_hsp08_pct0/MOTOR'
path = '{}/model.pt'.format(base_path)
mat = get_assigned_mat(path)

name_dict = {1: 'Visual', 2: 'Somatomotor', 3: 'Dorsal Attention', 4: 'Ventral Attention', 5: 'Limbic', 6: 'Frontoparietal', 7: 'Default'}
f, axes = plt.subplots(8, 8, figsize=(25, 22), subplot_kw={'xticks': [], 'yticks': []})
f.subplots_adjust(hspace=0.0001, wspace=0.001)
ax = axes.flat
max_val = mat.max()
mat[mat == 0] = None
for i in range(61):
    im = ax[i].imshow(mat[:, :, i], origin='lower', cmap='jet', vmin=0,
                      vmax=max_val)
    ax[i].axis('off')
f.patch.set_visible(False)
mask[mask == 0] = None
# im = plt.imshow(f_nets[:, :, 25], interpolation='none', cmap='jet', vmin=0, vmax=7)
plt.axis('off')
f.savefig(base_path + '/white_background.png')