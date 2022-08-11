import scipy.stats as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib
matplotlib.rc('axes.formatter', useoffset=False)

model_layer = torch.load('/users/nivl/data/autoencoder/hsp/hsp05_50pct2_masking_layer/models/epoch_1000.pt')
model_node = torch.load('/users/nivl/data/autoencoder/hsp/hsp05_50pct1_masking_node/models/epoch_1000.pt')


def calculate_hsp(w):
    n_nodes = w.shape
    num_elements = n_nodes

    # Calculate NZR and HSP sparsness
    norm_ratio = LA.norm(w, 1) / LA.norm(w, 2)
    h = (np.sqrt(num_elements) - norm_ratio.item()) / (np.sqrt(num_elements) - 1)

    return h


layer_w = model_layer['encoder.weight'].cpu().numpy()
node_w = model_node['encoder.weight'].cpu().numpy()

layer_hsp = [st.kurtosis(layer_w[i]) for i in range(100)]
node_hsp = [st.kurtosis(node_w[i]) for i in range(100)]
#
# layer_hsp = [calculate_hsp(layer_w[i]) for i in range(100)]
# node_hsp = [calculate_hsp(node_w[i]) for i in range(100)]


ax1 = plt.subplot(211)
ax1.set_title('Layer-wise')
ax1.set_ylim([-2, 50])
ax1.plot(range(100), layer_hsp)

ax2 = plt.subplot(212, sharex=ax1)
ax2.set_title('Node-wise')
ax2.plot(range(100), node_hsp)

plt.show()