import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from datetime import datetime as dt
import random
from data_utils import get_sbj_task_data, is_subject_valid, get_task_retest_data, convert_test_data
import timeit
import itertools
import collections
import scipy.io as sio
import pickle
import shutil

# path_dict = {'ft_rbm': 'finetune/0916/133819_hsp08_pct0_rbm/WM', 'ft_ae': 'finetune/0907/075346_hsp08_pct0/WM',
#              'random': 'clean/0911/112119_hsp08_pct0/WM', 'fx_rbm': 'fixed/0820/112202_WM_hsp08_pct0_dbn/l1_0.0005_l2_0.0005', 'fx_ae': 'fixed/0901/073348_WM_hsp08_pct0_ae/l1_5e-05_l2_0.0005'}
#
# acc_dict = {}
# col_dict = {'ft_rbm': '#1f77b4', 'ft_ae': '#ff7f0e',
#              'random': '#2ca02c', 'fx_rbm': '#d62728', 'fx_ae': '#9467bd'}

