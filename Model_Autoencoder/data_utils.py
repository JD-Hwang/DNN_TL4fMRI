import numpy as np
import random
import pickle
import scipy.io as sio
import scipy.stats as stats
import torch
import os
import nibabel as nib
import math

tasks = {'WM': ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools'],
         'MOTOR': ['lh', 'rh', 'lf', 'rf', 't'], 'EMOTION': ['neut', 'fear'], 'RELATIONAL': ['relation', 'match'],
         'SOCIAL': ['mental', 'rnd'], 'LANGUAGE': ['math', 'story'], 'GAMBLING': ['win', 'loss']}

retest_sbj = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', '137128',
              '139839', '144226', '146129', '149337', '149741', '151526', '158035', '169343', '172332',
              '175439', '177746', '185442', '187547', '192439', '194140', '195041', '200109', '200614',
              '204521', '250427', '287248', '433839', '562345', '599671', '601127', '627549', '660951',
              '662551', '783462', '859671', '861456', '877168', '917255']


def add_noise(img, in_dim, noise_type="masking", corruption_level=0.5):
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        mean = 0
        var = 0.5
        sigma = var ** .5
        noise = np.random.normal(size=img.shape) * sigma
        # noise = noise.reshape(row, col)
        img = img + noise
        return img

    elif noise_type == "speckle":
        noise = np.random.randn(in_dim)
        # noise = noise.reshape(row, col)
        img = img + img * noise
        return img

    elif noise_type == 'masking':
        num_corrupt = int(corruption_level * in_dim)
        corrupt_ids = random.sample(range(in_dim), num_corrupt)
        img[corrupt_ids] = 0
        return img

    else:
        print("BAD NOISE TYPE")


def get_data(subjects, in_dim, pre_loaded, denoising, pct):
    data = None
    if subjects == 100:
        for mat_idx in range(1, 11):
            matfile = sio.loadmat('/users/jmy/data/hcp_data/new_input2/sbj10_{}.mat'.format(mat_idx))
            mat = matfile['x_{}'.format(mat_idx)]
            if data is None:
                data = np.vstack(mat)
            else:
                data = np.concatenate((data, np.vstack(mat)))
        zscored = stats.zscore(data, ddof=1)

    else:
        if pre_loaded:
            f = open('/users/nivl/code/DAE/data/all/masking/clean_data.p', 'rb')
            zscored = pickle.load(f)['samples']
            f.close()
        else:
            matfile = sio.loadmat('/users/jmy/data/hcp_data/new_input2/sbj10_{}.mat'.format(3))
            mat = matfile['x_{}'.format(3)]
            stacked_data = np.vstack(mat)
            zscored = stats.zscore(stacked_data, ddof=1)

    samples = torch.from_numpy(zscored).type(torch.FloatTensor)

    noisy_samples = np.zeros((len(zscored), in_dim)).astype(np.float32) if denoising else None
    if denoising:
        if pre_loaded:
            f = open('/users/nivl/code/DAE/data/all/masking/masked_data_{}pct.p'.format(pct), 'rb')
            traindata = pickle.load(f)['traindata']
            f.close()
            noisy_samples = torch.from_numpy(traindata).type(torch.FloatTensor)
        else:
            corruption_level = pct / 100
            for idx in range(len(zscored)):
                noisy_samples[idx] = add_noise(zscored[idx], in_dim, noise_type='masking', corruption_level=corruption_level)
            noisy_samples = torch.from_numpy(noisy_samples).type(torch.FloatTensor)

    return samples, noisy_samples


def is_valid(sbj):
    filepath = '/data4/open_data/HCP/{}/MNINonLinear/Results/rfMRI_REST1_LR/vec1d.mat'.format(sbj)
    if os.path.isfile(filepath):
        return True

    return False


def get_sbj_data(sbj_id):
    sbj_data = None
    for scan in [1, 2]:
        for phase in ['LR', 'RL']:
            filepath = '/data4/open_data/HCP/{}/MNINonLinear/Results/rfMRI_REST{}_{}/vec1d.mat'.format(sbj_id, scan, phase)
            if os.path.isfile(filepath):
                f = sio.loadmat(filepath)
                mat = f['x_1d']
                if sbj_data is None:
                    sbj_data = mat
                else:
                    sbj_data = np.concatenate((sbj_data, mat))
    return sbj_data


def collect_sbj_data(subjects):
    path = '/data4/open_data/HCP'
    all_sbj = [x for x in os.listdir(path) if x.isdigit() and x not in retest_sbj]
    grp_data = None

    if subjects == 'all':
        grp_data = np.zeros((3896858, 52470), dtype=np.float32)
        start_idx = 0
        for idx, sbj_id in enumerate(all_sbj):
            sbj_data = get_sbj_data(sbj_id)
            if sbj_data is None:
                continue
            end_idx = start_idx + sbj_data.shape[0]
            grp_data[start_idx:end_idx, :] = sbj_data
            start_idx = end_idx
            print(idx)
    else:
        random.shuffle(all_sbj)
        num_sbj = 0
        sbj_idx = 0

        while num_sbj < subjects:
            sbj = all_sbj[sbj_idx]
            if not is_valid(sbj):
                sbj_idx += 1
                continue
            f = sio.loadmat('{}/{}/MNINonLinear/Results/rfMRI_REST1_LR/vec1d.mat'.format(path, sbj))
            mat = f['x_1d']
            if grp_data is None:
                grp_data = mat
            else:
                grp_data = np.concatenate((grp_data, mat))
            num_sbj += 1
            sbj_idx += 1

    grp_data = stats.zscore(grp_data, axis=0, ddof=1)
    grp_data = torch.from_numpy(grp_data)

    return grp_data


def get_block_samples(block_paths):
    block_data = None
    for path in block_paths:
        f = sio.loadmat(path)
        mat = f['x_1d']
        if block_data is None:
            block_data = mat
        else:
            block_data = np.concatenate((block_data, mat))
    samples = stats.zscore(block_data, axis=0, ddof=1)
    samples = torch.from_numpy(samples)

    return samples


def get_retest_data(sbj_id, mode):
    samples = None
    for scan in [1, 2]:
        for phase in ['LR', 'RL']:
            if mode == 'test':
                sbj_path = '/data4/open_data/HCP/{}/MNINonLinear/Results/rfMRI_REST{}_{}/vec1d.mat'.format(sbj_id, scan, phase)
            else:
                sbj_path = '/data4/open_data/HCP/test_retest/REST{}/{}/MNINonLinear/Results/rfMRI_REST{}_{}/vec1d.mat'.format(scan, sbj_id, scan, phase)

            f = sio.loadmat(sbj_path)
            mat = f['x_1d']

            if samples is None:
                samples = mat
            else:
                samples = np.concatenate((samples, mat))

    samples = stats.zscore(samples, axis=0, ddof=1)
    samples = torch.from_numpy(samples)
    return samples


def get_assigned_mat(path, epoch):
    img = nib.load('/users/nivl/code/DAE/data/flipped_hcp_8xx.nii')
    mask = img.get_fdata()
    idx = np.where(mask.flatten() == 1)[0]
    model = torch.load(path + '/models/epoch_{}.pt'.format(epoch))
    layer_list = []

    for key, value in model.items():
        if 'weight' in key:
            layer_list.append(value.cpu().numpy())

    tmp = layer_list[0]

    zscored = stats.zscore(tmp, axis=None, ddof=1)

    thrshed = zscored.copy()
    i, j = np.where(thrshed < 1.96)
    thrshed[i, j] = 0

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


def is_subject_valid(subject, task, retest=False):
    if not retest and subject in retest_sbj:
        return False

    for v in tasks[task]:
        filename = 'vol1d_{}_babi_python.mat'.format(v)
        lr = '/data4/open_data/HCP/{}/MNINonLinear/Results/tfMRI_{}_LR/{}'.format(subject, task, filename)
        rl = '/data4/open_data/HCP/{}/MNINonLinear/Results/tfMRI_{}_RL/{}'.format(subject, task, filename)
        if not os.path.isfile(lr) or not os.path.isfile(rl):
            return False

        if retest:
            lr = '/data4/open_data/HCP/test_retest/tfMRI_{}/{}/MNINonLinear/Results/tfMRI_{}_LR/{}'.format(task, subject, task, filename)
            rl = '/data4/open_data/HCP/test_retest/tfMRI_{}/{}/MNINonLinear/Results/tfMRI_{}_RL/{}'.format(task, subject, task, filename)
            if not os.path.isfile(lr) or not os.path.isfile(rl):
                return False

    return True


def get_sbj_task_data(sbj_id, task, retest=False):
    sbj_samples = None
    sbj_labels = None
    for idx, cond in enumerate(tasks[task]):
        filename = 'vol1d_{}_babi_python.mat'.format(cond)
        if retest:
            lr_file = '/data4/open_data/HCP/test_retest/tfMRI_{}/{}/MNINonLinear/Results/tfMRI_{}_LR/{}'.format(task,
                                                                                                                sbj_id,
                                                                                                                task,
                                                                                                                filename)
            rl_file = '/data4/open_data/HCP/test_retest/tfMRI_{}/{}/MNINonLinear/Results/tfMRI_{}_RL/{}'.format(task,
                                                                                                                sbj_id,
                                                                                                                task,
                                                                                                                filename)
        else:
            lr_file = '/data4/open_data/HCP/{}/MNINonLinear/Results/tfMRI_{}_LR/{}'.format(sbj_id, task, filename)
            rl_file = '/data4/open_data/HCP/{}/MNINonLinear/Results/tfMRI_{}_RL/{}'.format(sbj_id, task, filename)

        lr = sio.loadmat(lr_file)
        rl = sio.loadmat(rl_file)
        lr_samples = lr['samples1d']
        rl_samples = rl['samples1d']

        cond_samples_64 = np.concatenate((lr_samples, rl_samples))
        cond_samples = np.float32(cond_samples_64)
        dim = cond_samples.shape[0]

        cond_labels = np.full(dim, idx)

        if sbj_samples is None:
            sbj_samples = cond_samples
            sbj_labels = cond_labels
        else:
            sbj_samples = np.concatenate((sbj_samples, cond_samples))
            sbj_labels = np.concatenate((sbj_labels, cond_labels))
    return sbj_samples, sbj_labels


def get_task_retest_data(task):
    samples = None
    labels = None
    for idx, sbj_id in enumerate(retest_sbj):
        print(idx)
        # if idx > 5:
        #     break
        for idx, cond in enumerate(tasks[task]):
            for phase in ['LR', 'RL']:
                for mode in ['test', 'retest']:
                    # if mode == 'retest':
                    #     continue
                    if mode == 'test':
                        sbj_path = '/data4/open_data/HCP/{}/MNINonLinear/Results/tfMRI_{}_{}/vol1d_{}_babi_python.mat'.format(sbj_id, task, phase, cond)
                    else:
                        task_dir = [s for s in os.listdir('/data4/open_data/HCP/test_retest') if s.endswith(task)][0]
                        sbj_path = '/data4/open_data/HCP/test_retest/{}/{}/MNINonLinear/Results/tfMRI_{}_{}/vol1d_{}_babi_python.mat'.format(task_dir, sbj_id, task, phase, cond)

                    if not os.path.isfile(sbj_path):
                        continue
                    mat = sio.loadmat(sbj_path)
                    mat_samples = mat['samples1d']
                    cond_samples = np.float32(mat_samples)
                    dim = cond_samples.shape[0]

                    cond_labels = np.full(dim, idx)

                    if samples is None:
                        samples = cond_samples
                        labels = cond_labels
                    else:
                        samples = np.concatenate((samples, cond_samples))
                        labels = np.concatenate((labels, cond_labels))
    # samples = stats.zscore(samples, axis=0, ddof=1)

    return samples, labels


def convert_test_data(samples, model):
    batch = 350
    n_batches = int(math.ceil(samples.shape[0] / batch))
    out_samples = None
    for i in range(n_batches):
        idx = range(i * batch, (i+1) * batch) if i != (n_batches - 1) else range(i * batch, samples.shape[0])
        samples_batch = samples[idx]
        samples_batch = samples_batch.cuda()
        out = model(samples_batch)

        if out_samples is None:
            out_samples = out
        else:
            out_samples = torch.cat((out_samples, out))

    return out_samples.cpu()


def get_sbj_data_alltask(sbj_id):

    task_cnt = 0
    sbj_samples = None
    sbj_labels = None
    for k, v in tasks.items():

        for cond in v:
            filename = 'vol1d_{}_babi_python.mat'.format(cond)
            lr_file = '/data4/open_data/HCP/{}/MNINonLinear/Results/tfMRI_{}_LR/{}'.format(sbj_id, k, filename)
            rl_file = '/data4/open_data/HCP/{}/MNINonLinear/Results/tfMRI_{}_RL/{}'.format(sbj_id, k, filename)

            lr = sio.loadmat(lr_file)
            rl = sio.loadmat(rl_file)
            lr_samples = lr['samples1d']
            rl_samples = rl['samples1d']
            samples_64 = np.concatenate((lr_samples, rl_samples))
            samples_32 = np.float32(samples_64)
            dim = samples_32.shape[0]

            task_labels = np.full(dim, task_cnt)

            if sbj_samples is None:
                sbj_samples = samples_32
                sbj_labels = task_labels

            else:
                sbj_samples = np.concatenate((sbj_samples, samples_32))
                sbj_labels = np.concatenate((sbj_labels, task_labels))

            task_cnt += 1

    return sbj_samples, sbj_labels
