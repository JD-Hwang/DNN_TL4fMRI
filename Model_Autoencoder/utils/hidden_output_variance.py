import torch
from data_utils import get_data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


in_dim = 59583
output_dim = 100
path = '/users/nivl/data/autoencoder/hsp/pre_1304/20200403_120506'


class TiedAutoEncoderOffTheShelf(nn.Module):
    def __init__(self, inp, out, weight1):
        super(TiedAutoEncoderOffTheShelf, self).__init__()
        self.encoder = nn.Linear(inp, out, bias=True)
        self.decoder = nn.Linear(out, inp, bias=True)

        # tie the weights
        self.encoder.weight.data = weight1.clone()
        nn.init.kaiming_normal_(self.encoder.weight)
        self.decoder.weight.data = self.encoder.weight.data.transpose(0, 1)

    def forward(self, x):
        encoded_feats = F.relu(self.encoder(x), inplace=True)
        reconstructed_output = self.decoder(encoded_feats)
        return reconstructed_output, encoded_feats


model = TiedAutoEncoderOffTheShelf(in_dim, output_dim, torch.randn(output_dim, in_dim))

params = torch.load(path +'/models/epoch_2000.pt')
model.load_state_dict(params)
model.cuda()
enc_weights = params['encoder.weight'].cpu().numpy()

samples, noisy_samples = get_data(10, in_dim, True, True, 30)

noisy_samples = noisy_samples.cuda()

all_outputs = np.zeros((100, 12000))

for i in range(0, 12000, 200):

    batch_input = noisy_samples[i: i + 200]
    _, enc_output = model(batch_input)
    enc_np = enc_output.detach().cpu().numpy()
    all_outputs[0:, i: i+200] = enc_np.T

out_var_list = []
weight_var_list = []
out_mean_list = []
weight_mean_list = []
for i in range(100):
    weight_var, weight_mean = np.std(enc_weights[i]), np.mean(enc_weights[i])
    out_var, out_mean = np.std(all_outputs[i]), np.mean(all_outputs[i])

    out_var_list.append(out_var)
    weight_var_list.append(weight_var)
    out_mean_list.append(out_mean)
    weight_mean_list.append(weight_mean)


plot = plt.figure()
plt.title('Encoder Output Variance', fontsize=20)
plt.bar(range(100), out_var_list)
plt.plot(range(100), out_mean_list, 'r')
plot.savefig('{}/out_var.png'.format(path))
plt.close()

plot = plt.figure()
plt.title('Hidden Node Weight Variance', fontsize=20)
plt.bar(range(100), weight_var_list)
plt.plot(range(100), weight_mean_list, 'r')
plot.savefig('{}/weight_var.png'.format(path))
plt.close()





