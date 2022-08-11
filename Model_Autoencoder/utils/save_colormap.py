import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches
from data_utils import get_assigned_mat
import pickle

fl = np.load('/users/nivl/code/DAE/data/flip_yeo7.npz')
f_nets = fl['yeo7']

# fl = np.load('/users/nivl/code/DAE/data/mask_fnets_intersect.npz')
# f_nets = fl['f_nets']
# mask = fl['mask']
img = nib.load('/users/nivl/code/DAE/data/flipped_hcp_8xx.nii')
mask = img.get_fdata()
path = '/users/nivl/data/autoencoder/hsp/all/hsp08_50pct'
# path = '/users/nivl/data/autoencoder/hsp/all/hsp05_nonepct'
mat = get_assigned_mat(path, 10)
# f_nets[mask != 0] = 0
# mat[f_nets != 0] = 0
values = np.unique(mat.ravel())[1:]
name_dict = {1: 'Visual', 2: 'Somatomotor', 3: 'Dorsal Attention', 4: 'Ventral Attention', 5: 'Limbic', 6: 'Frontoparietal', 7: 'Default'}
f, axes = plt.subplots(8, 8, figsize=(25, 22), subplot_kw={'xticks': [], 'yticks': []})
f.subplots_adjust(hspace=0.0001, wspace=0.001)
ax = axes.flat
# for i in range(61):
#     im = ax[i].imshow(mat[:, :, i], origin='lower', cmap='jet', vmin=mat.min(),
#                       vmax=mat.max())
im = plt.imshow(f_nets[:, :, 25], interpolation='none', cmap='jet', vmin=mat.min(), vmax=mat.max())
colors = [im.cmap(im.norm(value)) for value in values]
# create a patch (proxy artist) for every color
# patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=5)) for i in range(len(values))]
# put those patched as legend-handles into the legend
# plt.legend(handles=patches, bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0.)
# vals = [3, 4, 5, 2, 6, 8, 9]
# plt.close()
# bar = plt.bar(range(7), vals, color=colors)
# plt.show()

color_dict = {}
for idx, val in enumerate(values):
    color_dict[val] = colors[idx]

pickle.dump(color_dict, open(path + "/color_dict.p", "wb"))
