import torch
import matplotlib.pyplot as plt
import numpy as np

path = '/users/nivl/data/autoencoder/hsp/20200417_154829_nosp'

model = torch.load(path + '/models/epoch_2000.pt')
b_enc = model['encoder.bias']
b_dec = model['decoder.bias']

b_enc = b_enc.cpu().numpy()
b_dec = b_dec.cpu().numpy()

plot = plt.figure(figsize=(16, 9))
plt.title('Encoder Bias Values', fontsize=20)
plt.plot(range(len(b_enc)), b_enc)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plot.savefig('{}/enc_bias.png'.format(path))
plt.close()

plot = plt.figure(figsize=(16, 9))
plt.title('Encoder Bias Histogram', fontsize=20)
plt.hist(b_enc, 50, alpha=0.75)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plot.savefig('{}/enc_hist.png'.format(path))
plt.close()

plot = plt.figure(figsize=(16, 9))
plt.title('Decoder Bias Values', fontsize=20)
plt.plot(range(len(b_dec)), b_dec)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plot.savefig('{}/dec_bias.png'.format(path))
plt.close()

plot = plt.figure(figsize=(16, 9))
plt.title('Decoder Bias Histogram', fontsize=20)
plt.hist(b_dec, 50, density=True, alpha=0.75)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plot.savefig('{}/dec_hist.png'.format(path))
plt.close()
