import nibabel as nib
import numpy as np
import os
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import torch

img = nib.load('/users/nivl/code/DAE/data/hcp_babi_msk95.nii')
mask = img.get_fdata()
idx = np.where(mask.flatten() == 1)[0]

# folder_path = '/users/nivl/data/autoencoder/hsp'
folder_path = '/users/nivl/data/autoencoder/hsp/2804'
paths = [x for x in os.listdir(folder_path)]
for path in paths:
    print(path)
    run_path = '{}/{}'.format(folder_path, path)
    run_files = os.listdir(run_path)

    epoch_dir = '{}/models'.format(run_path)
    ax_dir = epoch_dir + '/ax10'
    os.mkdir(ax_dir)
    epoch_models = [x for x in os.listdir(epoch_dir) if x.endswith('.pt')]
    for model_f in epoch_models:
        epoch = model_f[6:-3]
        print(epoch)
        model = torch.load(epoch_dir + '/{}'.format(model_f))

        layer_list = []

        for key, value in model.items():
            if 'weight' in key:
                layer_list.append(value.cpu().numpy())

        tmp = layer_list[0]

        nodes_n = tmp.shape[0]
        zscored = stats.zscore(tmp, axis=1, ddof=1)
        zscored = stats.zscore(zscored, axis=0, ddof=1)

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
        f.suptitle('Epoch: {}'.format(epoch), fontsize=100)
        for i in range(61):
            im = ax[i].imshow(assignmax[:, :, i], origin='lower', cmap='jet', vmin=assignmax.min(), vmax=assignmax.max())
        plt.savefig('{}/epoch_{}.png'.format(ax_dir, epoch))
        plt.close()
