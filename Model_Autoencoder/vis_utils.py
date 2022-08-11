import nibabel as nib
import numpy as np
import os
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import torch


def save_map(path, epoch, method='zval'):
    img = nib.load('/users/nivl/code/DAE/data/flipped_hcp_8xx.nii')
    mask = img.get_fdata()
    idx = np.where(mask.flatten() == 1)[0]

    out_dir = path + '/maps'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = torch.load(path + '/epoch_{}.pt'.format(epoch))
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
    f, axes = plt.subplots(8, 8, figsize=(25, 22), subplot_kw={'xticks': [], 'yticks': []})
    f.subplots_adjust(hspace=0.0001, wspace=0.001)
    ax = axes.flat
    for i in range(61):
        im = ax[i].imshow(assignmax[:, :, i], origin='lower', cmap='jet', vmin=assignmax.min(),
                          vmax=assignmax.max())
    plt.savefig('{}/epoch_{}_{}.png'.format(out_dir, epoch, method))
    plt.close()


def plot_map(map, vmin, vmax, path):
    f, axes = plt.subplots(8, 8, figsize=(25, 22), subplot_kw={'xticks': [], 'yticks': []})
    f.subplots_adjust(hspace=0.0001, wspace=0.001)
    ax = axes.flat
    for i in range(61):
        im = ax[i].imshow(map[:, :, i], origin='lower', cmap='jet', vmin=vmin,
                          vmax=vmax)
    plt.savefig(path)
    plt.close()
