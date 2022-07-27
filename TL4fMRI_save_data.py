### data save functions
import os
import numpy as np
from pytz import timezone
from datetime import datetime as dt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

class save_data:

    def __init__(self,config):
        self.config = config
        self.target_task = config['task_list'][config['target_task']]
        
    ### Set save file path
    def save_file(self,p_model,tgs):

        root_path = os.getcwd()
        save_path = root_path + "/Transfer_learning/"
        output_folder = save_path + "{}/".format(self.target_task) + "{}_{}-{}".format(p_model[0],tgs[0],tgs[1])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print("Save file path : " + str(output_folder))

        return output_folder
    
    ### Make save directory for each parameter candidates including Pretrained model, Target sparsity
    def make_sp_folder(self,p_model,tgs):
        model_name = p_model[0]
        tg_hsp = "+tgs:{}_{}+".format(p_model[1],tgs[0]) #target sparsity
        lr = "lr:{}+".format(self.config['learning_rate']) #learning rate
        max_b = "max_b:{}+".format(self.config['hsc']['max_b']) #max beta
        batch_size = "batch:"+str(self.config['batch_size'])+"+"
        mome = "momentum:"+str(self.config['momentum'])+"+"
        rand = "seed:"+str(self.config['random_state'])

        tgs_folder = self.save_file(p_model,tgs)+"/"+model_name+tg_hsp+lr+max_b+batch_size+mome+rand
        os.mkdir(tgs_folder)
        return tgs_folder
    
    ### Save parameter text file
    def save_parameters(self,p_model,save_lr_path):

        tz = timezone('Asia/Seoul')
        year = dt.now(tz).year
        month = '0{}'.format(dt.now(tz).month) if dt.now(tz).month < 10 else str(dt.now(tz).month)
        day = '0{}'.format(dt.now(tz).day) if dt.now(tz).day < 10 else str(dt.now(tz).day)
        hour = '0{}'.format(dt.now(tz).hour) if dt.now(tz).hour < 10 else str(dt.now(tz).hour)
        minute = '0{}'.format(dt.now(tz).minute) if dt.now(tz).minute < 10 else str(dt.now(tz).minute)
        sec = '0{}'.format(dt.now(tz).second) if dt.now(tz).second < 10 else str(dt.now(tz).second)

        f = open(save_lr_path + "/training_parameters.txt", 'w')
        f.write("Started time : {}{}{}_{}:{}".format(year, month, day, hour, minute) + '\n')
        f.write('Task : ' + str(self.target_task) + '\n')
        f.write('Model : '+str(p_model)+ '\n')
        f.write('Train subjects : ' + str(self.config['train_n_subject']) + '\n')
        f.write('Test subjects : ' + str(self.config['test_n_subject']) + '\n')
        f.write('Node combination : ' + str(self.config['comb']['comb_num']) + '\n')
        f.write('Epoch : ' + str(self.config['num_epoch']) + '\n')
        f.write('Batch size : ' + str(self.config['batch_size']) + '\n')
        f.write('Activation : ' + str(self.config['activation']) + '\n')
        f.write('Activation2 : ' + str(self.config['activation2']) + '\n')
        f.write('Learning rate : ' + str(self.config['learning_rate']) + '\n')
        f.write("Momentum : " + str(self.config['momentum']) + '\n')
        f.write('Dropout rate : ' + str(self.config['dropout_rate']) + '\n')
        f.write('HSP type : ' + str(self.config['hsc']['type']) + '\n')
        f.write('Max beta : ' + str(self.config['hsc']['max_b']) + '\n')
        f.write('Beta learning rate : ' + str(self.config['hsc']['b_lr']) + '\n')
        f.write('EarlyStopping_patience : ' + str(self.config['es_patience']) + '\n')
        f.write('LearningRate_patience : ' + str(self.config['lr_patience']) + '\n')
        f.write('LearningRate_anneal_factor : ' + str(self.config['anneal_factor']) + '\n')
        f.write('Num of fold : ' + str(self.config['num_fold']) + '\n')
        f.write('Randomseed : ' + str(self.config['random_state']) + '\n')
        f.close()

        return
    
    ### Save plots to visualize training status for each epochs
    def save_plot_during_training(self,p_model,tgs,save_lr_path,hsp_list,fold,epoch,outer_train_acc,outer_valid_acc,outer_test_acc,outer_lr_list,model_epoch):

        label1 = mpatches.Patch(color='magenta', label='layer1-hsp')
        label2 = mpatches.Patch(color='blue', label='layer2-hsp')
        label3 = mpatches.Patch(color='gray', label='Train error')
        label4 = mpatches.Patch(color='orange', label='Validation error')
        label5 = mpatches.Patch(color='red', label='Test error')
        label6 = mpatches.Patch(color='green', label='Learning rate')

        with sns.axes_style("whitegrid"):
            fig,ax = plt.subplots(figsize=(15,8))
            if 'Fix' in p_model[0]:
                plt.title("ACC&HSP, {}-({}-{})-fold{}".format(p_model[0],p_model[1],tgs,fold+1))
                ax.plot(np.array(hsp_list)[:epoch+1], c = 'm', linestyle = '--')
                plt.legend(handles = [label2,label3,label4,label5,label6])
            else:
                plt.title("ACC&HSP, {}-({}-{})-fold{}".format(p_model[0],p_model[1],tgs[0],fold+1))
                ax.plot(np.array(hsp_list)[:epoch+1,0], c = 'm', linestyle = '--')
                ax.plot(np.array(hsp_list)[:epoch+1,1], c = 'b', linestyle = '--')
                plt.legend(handles = [label1,label2,label3,label4,label5,label6])
            ax.set_ylabel("Hoyer's sparsity")
            ax2 = ax.twinx()
            ax2.plot(100-np.array(outer_train_acc), c='gray')
            ax2.plot(100-np.array(outer_valid_acc), c='orange')
            ax2.plot(100-np.array(outer_test_acc), c='red')
            ax2.set_ylabel("Error rate")

            ax3=ax.twinx()
            ax3.plot(outer_lr_list,c ='g')
            ax3.axes.get_yaxis().set_ticks([])

            if model_epoch !=0:
                ax.axvline(x=model_epoch, c='olive')

          #  plt.legend(handles = [label1,label2,label3,label4,label5,label6])

            plt.savefig(save_lr_path+"/learning_curve_{}fold".format(fold+1))
            plt.close()
        return
