import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import hdf5storage

path = '/users/nivl/data/autoencoder/nonsparse/20200218_115844'
features_3d = '{}/features_3d.mat'.format(path)

features1 = sio.loadmat(features_3d)['features1']
features2 = sio.loadmat(features_3d)['features2']
data = np.concatenate((features1, features2), axis=0)
data[data < 1.96] = 0

overlap_vol = np.zeros((61, 73, 61))
for (x,y,z), value in np.ndenumerate(data[0]):
    vox_lst = [data[j, x, y, z] for j in range(500)]
    sig = np.max(vox_lst)
    if sig > 0:
        sig_i = vox_lst.index(sig)
        overlap_vol[x][y][z] = sig_i + 1

f, axes = plt.subplots(8, 8, figsize=(25, 22), subplot_kw={'xticks': [], 'yticks': []})
f.subplots_adjust(hspace=0.0001, wspace=0.001)
ax = axes.flat
for k in range(61):
    im = ax[k].imshow(overlap_vol[:, :, k], origin='lower', cmap='jet', vmin=overlap_vol.min(), vmax=overlap_vol.max())
plt.savefig('{}/weight_map.png'.format(path))
