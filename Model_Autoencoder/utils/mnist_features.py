import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os

path = '/users/nivl/data/autoencoder/mnist/20200417_163100_sgd'
# os.makedirs(path + '/features')


def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = x.view(1, 28, 28)
    return x


model = torch.load(path + '/models/epoch_5000.pth')
w_enc = model['encoder.weight']

w_enc = w_enc.cpu().data

for i in range(100):
    pic = to_img(w_enc[i])
    save_image(pic, '{}/features/image_{}.png'.format(path, i))