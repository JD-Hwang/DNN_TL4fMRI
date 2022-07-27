### Hoyer sparsity control
import torch
import numpy as np

class HSP:
    def __init__(self,config,wsc_flag,p_model):
        self.config = config
        self.wsc_flag = wsc_flag
        self.pretrain_model = p_model[0]
    # Outputs: Hoyer sparsity level, beta 
    # Args: mode, weight, current beta, max_beta, beta_lr, target HSP, epsilon (set None) 

    def Hoyers_sparsity_control(self,mode, W, beta, max_b, beta_lr, tg_hsp): 
        from numpy import linalg as LA

        if mode=='layer':

            # Weight sparsity control with Hoyer's sparsness (Layer wise)  
            beta = beta

            # Get value of weight
            [n_nodes,dim]=W.shape  
            num_elements=dim*n_nodes

            Wvec=W.detach().flatten()

            # Calculate L1 and L2 norm    
            L1=torch.norm(Wvec,1,dim=0)
            L2=torch.norm(Wvec,2,dim=0)

            # Calculate hoyer's sparsness
            h=(np.sqrt(num_elements)-(L1/L2))/(np.sqrt(num_elements)-1)
            # Update beta
            beta-=beta_lr*torch.sign(h-tg_hsp)

            # Trim value
            beta=0.0 if beta<0.0 else beta
            beta=max_b if beta>max_b else beta

            return [h,beta]


        elif mode=='node':   ###

            # Weight sparsity control with Hoyer's sparsness (Node wise)
            b_vec = beta.to(device)
            W_np = W.detach() #cpu().detach().numpy()
            # Get value of weight
            [n_nodes,dim]=W_np.shape   #[5000,52470]

            # Calculate L1 and L2 norm
            L1=torch.norm(W_np,1,dim=1)
            L2=torch.norm(W_np,2,dim=1)

            h_vec = torch.zeros((1,n_nodes))
            tg_vec = (torch.ones(n_nodes)*tg_hsp).to(device)

            # Calculate hoyer's sparsness
            h_vec=(np.sqrt(dim)-(L1/L2))/(np.sqrt(dim)-1) #tensor

            # Update beta
            b_vec-=beta_lr*torch.sign(h_vec-tg_vec)

            # Trim value
            b_vec[b_vec>max_b]=max_b
            b_vec[tg_vec-h_vec<=0]=0.0

            return [h_vec,b_vec]

    # Sparsity control
    def sparsity_control(self,model, hsp_val, beta_val, hsp_list, beta_list, tg_hsp):

        l1_reg = None
        layer_idx = 0
        max_beta = self.config['hsc']['max_b']
        beta_lr = self.config['hsc']['b_lr']

        for name, temp_w in model.named_parameters():
            # Fixed models
            if 'Fix' in self.pretrain_model[0]:
                max_beta = self.config['hsc']['max_b'][1]
                beta_lr = self.config['hsc']['b_lr'][1]
                if "fc"  in name and "weight" in name and '2' not in name: # Control sparsity of fc1, fc2 layer
                    if self.wsc_flag[1] != 0:
                        if self.config['hsc']['type']=='layer':
                            hsp_val, beta_val = self.Hoyers_sparsity_control('layer',
                                temp_w, beta_val, max_beta, 
                                beta_lr, tg_hsp)

                            #Calcuate L1 norm
                            l1_val = torch.norm(temp_w, 1,dim=0)
                            #Calculate L1 norm with Hoyers sparsity control applied
                            layer_reg = torch.abs(l1_val) * beta_val

                        if self.config['hsc']['type']=='node':
                            hsp_val, beta_val = self.Hoyers_sparsity_control('node',
                                temp_w, beta_val, max_beta, 
                                beta_lr, tg_hsp)

                            #Calcuate L1 norm
                            l1_val = torch.norm(temp_w, 1,dim=1)
                            #Calculate L1 norm with Hoyers sparsity control applied
                            layer_reg = torch.abs(l1_val) * beta_val.clone().detach()
                    if l1_reg is None:
                        l1_reg = torch.sum(layer_reg)
                    else:
                        l1_reg = l1_reg + torch.sum(layer_reg)
                    layer_idx += 1

            else:    #Finetune models and Random Initialized models
                if "fc"  in name and "weight" in name and '3' not in name: # Control sparsity of fc1, fc2 layer
                    if self.wsc_flag[layer_idx] != 0:
                    #you can change wsc_flag to specify layer to control sparsity
                        if self.config['hsc']['type']=='layer':
                            hsp_val[layer_idx], beta_val[layer_idx] = self.Hoyers_sparsity_control('layer',
                                temp_w, beta_val[layer_idx], max_beta[layer_idx], 
                                beta_lr[layer_idx], tg_hsp[layer_idx])

                            #Calcuate L1 norm
                            l1_val = torch.norm(temp_w, 1,dim=0)
                            #Calculate L1 norm with Hoyers sparsity control applied
                            layer_reg = torch.abs(l1_val) * beta_val[layer_idx]

                        if self.config['hsc']['type']=='node':
                            hsp_val[layer_idx], beta_val[layer_idx] = self.Hoyers_sparsity_control('node',
                                temp_w, beta_val[layer_idx], max_beta[layer_idx], 
                                beta_lr[layer_idx], tg_hsp[layer_idx])

                            #Calcuate L1 norm
                            l1_val = torch.norm(temp_w, 1,dim=1)
                            #Calculate L1 norm with Hoyers sparsity control applied
                            layer_reg = torch.abs(l1_val) * beta_val[layer_idx].clone().detach()

                    if l1_reg is None:
                        l1_reg = torch.sum(layer_reg)
                    else:
                        l1_reg = l1_reg + torch.sum(layer_reg)
                    layer_idx += 1

        return l1_reg,hsp_val,beta_val

    #Initiate sparsity records
    def init_hsp(self):
        
        if self.config['hsc']['type']=='layer':
            hsp_val = [0,0]
            beta_val = hsp_val.copy()
            hsp_list = []
            beta_list = []
            for i in range(0,self.config['num_epoch']):
                hsp_list.append([0, 0])
                beta_list.append([0, 0])

        if self.config['hsc']['type']=='node':
            hsp_val = [torch.zeros(5000), torch.zeros(self.config['comb']['comb_num'][1])]
            beta_val = hsp_val.copy()
            hsp_list = []
            beta_list = []
            for i in range(0,self.config['num_epoch']):
                hsp_list.append([torch.zeros(5000), torch.zeros(self.config['comb']['comb_num'][1])])
                beta_list.append([torch.zeros(5000), torch.zeros(self.config['comb']['comb_num'][1])])

        return hsp_val, beta_val, hsp_list, beta_list
