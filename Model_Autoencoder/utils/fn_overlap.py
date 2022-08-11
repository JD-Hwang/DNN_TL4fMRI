import nibabel as nib
import numpy as np
import os
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import torch

net_dict = {1: 'Visual', 2: 'Somatomotor', 3: 'Dorsal Attention', 4: 'Ventral Attention', 5: 'Limbic', 6: 'Frontoparietal', 7: 'Default'}
img = nib.load('/users/nivl/code/DAE/data/hcp_babi_msk95.nii')
mask = img.get_fdata()
idx = np.where(mask.flatten() == 1)[0]

# f = np.load('/users/nivl/code/DAE/data/flip_yeo7.npz')
# f_nets = f['yeo7']

fl = np.load('/users/nivl/code/DAE/data/mask_fnets_intersect.npz')
f_nets = fl['f_nets']

base_path = '/users/nivl/data/autoencoder/hsp/all'

for run in os.listdir(base_path):
    if run != 'hsp08_50pct':
        continue
    folder_path = '{}/{}'.format(base_path, run)

    model = torch.load(folder_path + '/models/epoch_10.pt')

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

    fn_count = 0
    inter_count = 0
    for (x,y,z), value in np.ndenumerate(assignmax):
        fn_val = f_nets[x][y][z]
        if fn_val > 0:
            fn_count += 1
            if value > 0:
                inter_count += 1

    f = open(folder_path + "/fn_overlap.txt", 'w')
    f.write('Overall: ' + str(inter_count / fn_count) + '\n')

    for net in range(1, 8):
        indices = zip(*np.where(f_nets == net))
        net_count = 0
        nodes_count = 0
        nodes_dict = {}
        for (x, y, z) in indices:
            net_count += 1
            node_val = assignmax[x][y][z]
            if node_val > 0:
                nodes_count += 1
                int_node = int(node_val)
                if int_node not in nodes_dict:
                    nodes_dict[int_node] = 1
                else:
                    nodes_dict[int_node] += 1

        ovr = nodes_count / net_count
        labels = []
        values = []
        for k, v in nodes_dict.items():
            labels.append(str(k))
            values.append(v / net_count)
        plot = plt.figure()
        plt.bar(range(len(labels)), values, align='center')
        plt.xticks(range(len(labels)), labels)
        plt.title('{0} Network: Overall overlap: {1:.3f}'.format(net_dict[net], ovr), fontsize=14)
        plt.xlabel('Nodes', fontsize=16)
        plt.ylabel('O(Node, FN)', fontsize=16)
        plot.savefig('{}/FN{}.png'.format(folder_path, net))
        plt.close()
        f.write('{} : {} \n'.format(net_dict[net], ovr))
    f.close()
