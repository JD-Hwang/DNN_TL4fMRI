### data save functions
import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio

class load_sbj:
    def __init__(self,pretrain_type_list):
        self.base_path = os.getcwd()

        self.task_dict = {
            'WM': ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools'],
            'MOTOR': ['lh', 'rh', 'lf', 'rf', 't'], 
            'EMOTION': ['neut', 'fear'], 
            'RELATIONAL': ['relation', 'match'], 
            'SOCIAL': ['mental', 'rnd'], 
            'LANGUAGE': ['math', 'story'], 
            'GAMBLING': ['win', 'loss'], 
        }

        self.tasks = ['WM', 'MOTOR', 'EMOTION', 'RELATIONAL', 'SOCIAL', 'LANGUAGE', 'GAMBLING']
        self.num_samples = [[544], [260], [223, 224], [216], [268], [300], [271]]
        self.retest_sbj = [
            '103818', '105923', '111312', '114823', '115320', '122317', '125525', '135528', 
            '137128', '139839', '144226', '146129', '149337', '149741', '151526', '158035', 
            '169343', '172332', '175439', '177746', '185442', '187547', '192439', '194140', 
            '195041', '200109', '200614', '204521', '250427', '287248', '433839', '562345', 
            '599671', '601127', '627549', '660951', '662551', '783462', '859671', '861456', 
            '877168', '917255',
        ]
        self.pretrain_type_list = pretrain_type_list

    ### Check if subject is test/retest and have both LR/RL encoding fMRI data
    def is_subject_valid(self,subject, task, retest=False):
        retest_sbj = self.retest_sbj
        if not retest and subject in retest_sbj:
            return False

        for v in self.task_dict[task]:
            if "Fix" in self.pretrain_type_list[0][0]:
                filename = 'vol1d_{}_babi_python_{}_{}.mat'.format(v,self.pretrain_type_list[0][0],str(self.pretrain_type_list[0][1]).replace(".",''))
            else: filename = 'vol1d_{}_babi_python.mat'.format(v)
            lr = self.base_path+'/Sample_data/Transfer_learning_sbj/{}/tfMRI_{}_LR/{}'.format(subject, task, filename)
            rl = self.base_path+'/Sample_data/Transfer_learning_sbj/{}/tfMRI_{}_RL/{}'.format(subject, task, filename)
            if not os.path.isfile(lr) or not os.path.isfile(rl):
                return False

            if retest:
                lr = self.base_path+'/Sample_data/Transfer_learning_sbj/test_retest/{}/tfMRI_{}_LR/{}'.format(subject, task, filename)
                rl = self.base_path+'/Sample_data/Transfer_learning_sbj/test_retest/{}/tfMRI_{}_RL/{}'.format(subject, task, filename)
                if not os.path.isfile(lr) or not os.path.isfile(rl):
                    return False
                if subject not in retest_sbj:
                    return False
        return True

    ### Get subject task data
    def get_sbj_task_data(self,sbj_id, task, retest=False):
        sbj_samples = None
        sbj_labels = None
        for idx, cond in enumerate(self.task_dict[task]):
            if "Fix" in self.pretrain_type_list[0][0]:
                filename = 'vol1d_{}_babi_python_{}_{}.mat'.format(cond,self.pretrain_type_list[0][0],str(self.pretrain_type_list[0][1]).replace(".",''))
            else: filename = 'vol1d_{}_babi_python.mat'.format(cond)

            if retest:
                lr_file = self.base_path+'/Sample_data/Transfer_learning_sbj/test_retest/{}/tfMRI_{}_LR/{}'.format(sbj_id, task, filename)
                rl_file = self.base_path+'/Sample_data/Transfer_learning_sbj/test_retest/{}/tfMRI_{}_RL/{}'.format(sbj_id, task, filename)
            else:
                lr_file = self.base_path+'/Sample_data/Transfer_learning_sbj/{}/tfMRI_{}_LR/{}'.format(sbj_id, task, filename)
                rl_file = self.base_path+'/Sample_data/Transfer_learning_sbj/{}/tfMRI_{}_RL/{}'.format(sbj_id, task, filename)

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
    
    ### Get All subject data and concat
    def load_data(self,config):
        samples_per_sbj = self.num_samples[config['target_task']]
        task = self.tasks[config['target_task']]
        hcp_sbj = [e for e in os.listdir(self.base_path+'/Sample_data/Transfer_learning_sbj/') if e.isdecimal()]
        np.random.seed(config['random_state'])
        np.random.shuffle(hcp_sbj)

        train_samples = []
        train_labels = []
        train_subjects = []

        test_samples = []
        test_labels = []
        test_subjects = []

        sbj_list = []
        for sbj in hcp_sbj:
            if len(sbj_list) >= config['train_n_subject']:
                break
            if not self.is_subject_valid(sbj, task, False):
                continue
            sbj_list.append(sbj)

        for sbj in tqdm(sbj_list):
            sbj_samples, sbj_labels = self.get_sbj_task_data(sbj, task)

            train_samples.append(sbj_samples)
            train_labels.append(sbj_labels)
            train_subjects.append(np.repeat(sbj, len(sbj_samples)))

        tr_sbj_num = len(sbj_list)
        train_final_id = len(train_samples)

        train_samples = np.concatenate(train_samples)
        train_labels = np.concatenate(train_labels)
        train_subjects = np.concatenate(train_subjects)

        if config['output_size'] ==2:  #Make it onehotencoding
            train_labels_onehot = np.zeros((train_labels.size, train_labels.max() + 1))
            train_labels_onehot[np.arange(train_labels.size), train_labels] = 1
            train_labels = train_labels_onehot.astype(int)

        print('Loaded {} training subjects'.format(tr_sbj_num))

        sbj_list = []
        for sbj in hcp_sbj:
            if len(sbj_list) >= config['test_n_subject']:
                break
            if not self.is_subject_valid(sbj, task, True):
                continue
            sbj_list.append(sbj)

        for sbj in tqdm(sbj_list):
            sbj_test_samples, sbj_test_labels = self.get_sbj_task_data(sbj, task)
            sbj_retest_samples, sbj_retest_labels = self.get_sbj_task_data(sbj, task, retest=True)

            sbj_samples = np.concatenate((sbj_test_samples, sbj_retest_samples))
            sbj_labels = np.concatenate((sbj_test_labels, sbj_retest_labels))

            test_samples.append(sbj_samples)
            test_labels.append(sbj_labels)
            test_subjects.append(np.repeat(sbj, len(sbj_samples)))

        ts_sbj_num = len(sbj_list)

        test_samples = np.concatenate(test_samples)
        test_labels = np.concatenate(test_labels)
        test_subjects = np.concatenate(test_subjects)

        if config['output_size'] ==2:  #Make it onehotencoding
            test_labels_onehot = np.zeros((test_labels.size, test_labels.max() + 1))
            test_labels_onehot[np.arange(test_labels.size), test_labels] = 1
            test_labels = test_labels_onehot.astype(int)

        print('Loaded {} test subjects'.format(ts_sbj_num))    

        return train_samples, train_labels, train_subjects, test_samples, test_labels, test_subjects, task
