import os
import scipy.io as sio
from vis_utils import save_map

# path = '/users/nivl/data/autoencoder/test_retest/hsp08_50pct'
#
# for run in ['185442', '187547', '192439']:
#     run_dir = '{}/{}'.format(path, run)
#     print(run)
#     for sess in ['test', 'retest']:
#         model_dir = '{}/{}/models'.format(run_dir, sess)
#         for matfile in os.listdir(model_dir):
#             if matfile == 'maps':
#                 continue
#             epoch = matfile[6:-3]
#             save_map(model_dir, epoch)


path = '/users/nivl/data/autoencoder/hsp/2705'

for run in os.listdir(path):
    print(run)
    model_dir = '{}/{}/models'.format(path, run)
    for matfile in os.listdir(model_dir):
        if matfile == 'maps':
            continue
        epoch = matfile[6:-3]
        save_map(model_dir, epoch)
