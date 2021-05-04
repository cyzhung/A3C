# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:22:03 2021

@author: 123
"""

from models import ActorCritic
import torch.multiprocessing as mp
import torch.optim as optim
from worker import Worker
from Brain import test
import torch
import os
torch.set_num_threads(1)
NUM_PROCESSES=10
device="cpu"
           
                        
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                
                
                
MODEL_NUMBER=17000
stage='3'
world='8'
LEVEL=world+'-'+stage
PATH='./model/{}/ActorCritic_{}_{}.pkl'.format(LEVEL,LEVEL,MODEL_NUMBER)
               



    

if __name__=='__main__':    
    global_model=ActorCritic()
    optimizer=SharedAdam(global_model.parameters(),lr=1e-4)
    if(os.path.exists(PATH)):
        print('Loaded Model')
        check_point=torch.load(PATH)
        global_model.load_state_dict(check_point['model_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
    
    global_model.share_memory()
    
    
    workers=[mp.Process(target=test,args=(global_model,device,world,stage,0.05))]
    

    [w.start() for w in workers]
    
    

    #workers.append(test_worker)
    
    [w.join() for w in workers]
    
    

    
    
    