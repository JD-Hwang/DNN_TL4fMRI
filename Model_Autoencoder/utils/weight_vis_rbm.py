import nibabel as nib
import numpy as np
import os
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt

img = nib.load('/users/nivl/code/DAE/data/hcp_babi_msk95.nii')
mask = img.get_fdata()
idx = np.where(mask.flatten() == 1)[0]

matfile = sio.loadmat('/users/nivl/data/transfer_learning/result/200414/1/rbm/1_2000epoc.mat')
tmp = matfile['tmp_var']

# matfile = sio.loadmat('/users/nivl/data/transfer_learning/result/200311/1/result_weight.mat')
# tmp = matfile['weight'][0][0].T

nodes_n = tmp.shape[0]
zscored = stats.zscore(tmp, axis=None)

thrshed = zscored.copy()
i, j = np.where(thrshed < 1.96)
thrshed[i, j] = 0

# max assign
vec_assigned = np.zeros(59583)
for i, position in enumerate(thrshed.transpose()):
    if np.sum(position) == 0:
        continue
    n = np.argmax(position)
    vec_assigned[i] = n+1

# mapping
vol = np.zeros((61*73*61))
vol[idx] = vec_assigned

assignmax = np.reshape(vol, [61, 73, 61])
f, axes = plt.subplots(8, 8, figsize=(25, 22),subplot_kw={'xticks': [], 'yticks': []})
f.subplots_adjust(hspace=0.0001, wspace=0.001)
ax = axes.flat
for i in range(61):
    im = ax[i].imshow(assignmax[:, :, i], origin='lower', cmap='jet', vmin=assignmax.min(), vmax=assignmax.max())
plt.savefig('/home/nivl/Pictures/rbm_results_1404_2000.png')
