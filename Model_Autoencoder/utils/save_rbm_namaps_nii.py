import torch
import numpy as np
import os
import scipy.stats as stats
import nibabel as nib
import shutil
import scipy.io as sio


def get_assigned_mat(path, method='zval'):
    img = nib.load('/users/nivl/code/DAE/data/flipped_hcp_8xx.nii')
    mask = img.get_fdata()
    idx = np.where(mask.flatten() == 1)[0]
    model = sio.loadmat(path)

    tmp = model['tmp_w']

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
mask = fl['mask']
img = nib.load('/users/nivl/code/DAE/data/flipped_hcp_8xx.nii')


f_name = '1_15epoc'
base_path = '/users/nivl/data/autoencoder/dbn/epoch15'
path = '{}/{}.mat'.format(base_path, f_name)

nii_dir = base_path + '/namap_nii'
if not os.path.exists(nii_dir):
    os.makedirs(nii_dir)

mat = get_assigned_mat(path, 'zval')
nodes = np.unique(mat)[1:]

for node in nodes:
    node_mat = mat.copy()
    node_mat[node_mat != node] = 0
    node_mat[node_mat != 0] = 1
    ni_img = nib.Nifti1Image(node_mat, img.affine)
    nib.save(ni_img, '{}/node_{}.nii'.format(nii_dir, node))
shutil.copy2('/users/nivl/code/DAE/data/wMNI_avg152T1.nii', nii_dir)
