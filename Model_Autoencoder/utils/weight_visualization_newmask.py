import nibabel as nib
import numpy as np
import os
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import torch

img = nib.load('/users/nivl/code/PyTorchNets/data/rfMRI/hcp_babi_msk95.nii')
mask = img.get_fdata()
idx = np.where(mask.flatten() == 1)[0]

# folder_path = '/users/nivl/data/autoencoder/hsp'
folder_path = '/users/nivl/data/dnn_results/DEBUG'
paths = [x for x in os.listdir(folder_path) if x.startswith('20200413_1741')]
for path in paths:
    print(path)
    run_path = '{}/{}'.format(folder_path, path)
    run_files = os.listdir(run_path)
    if 'model.pt' not in run_files:
        epoch_dir = '{}/models'.format(run_path)
        epoch_models = os.listdir(epoch_dir)
        epochs = [int(x[6:-3]) for x in epoch_models]
        epochs.sort()
        model = torch.load(epoch_dir + '/epoch_{}.pt'.format(epochs[-1]))
    else:
        model = torch.load(run_path + '/model.pt')

    layer_list = []

    for key, value in model.items():
        if 'weight' in key:
            layer_list.append(value.cpu().numpy())

    tmp = layer_list[0]

    nodes_n = tmp.shape[0]
    zscored = stats.zscore(tmp, axis=None, ddof=1)

    thrshed = zscored.copy()
    i, j = np.where(thrshed < 1.96)
    thrshed[i, j] = 0

    # max assign
    vec_assigned = np.zeros((59583))
    for i, position in enumerate(thrshed.transpose()):
        if np.sum(position) == 0:
            continue
        n = np.argmax(position)
        vec_assigned[i] = n + 1

    # mapping
    vol = np.zeros((61*73*61))
    vol[idx] = vec_assigned

    assignmax = np.reshape(vol, [61, 73, 61])
    f, axes = plt.subplots(8, 8, figsize=(25, 22), subplot_kw={'xticks': [], 'yticks': []})
    f.subplots_adjust(hspace=0.0001, wspace=0.001)
    ax = axes.flat
    for i in range(61):
        im = ax[i].imshow(assignmax[:, :, i], origin='lower', cmap='jet', vmin=assignmax.min(), vmax=assignmax.max())
    plt.savefig('{}/weight_map_axnone_pos.png'.format(run_path))
    plt.close()
