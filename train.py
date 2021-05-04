# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:22:03 2021

@author: 123
"""

from models import ActorCritic
import torch.multiprocessing as mp
import torch.optim as optim
from worker import Worker
from utility import test
from ShareAdam import SharedAdam
import torch
import os
import argparse
torch.set_num_threads(1)
NUM_PROCESSES=10
device="cpu"

                        

                
         
                
MODEL_NUMBER=0
stage='3'
world='8'
LEVEL=world+'-'+stage
PATH='./model/{}/ActorCritic_{}.pkl'.format(LEVEL,MODEL_NUMBER)
               

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args=get_args()

    global_model=ActorCritic()
    optimizer=SharedAdam(global_model.parameters(),lr=1e-4)
    if(os.path.exists(PATH)):
        print('Loaded Model')
        check_point=torch.load(PATH)
        global_model.load_state_dict(check_point['model_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
    
    global_model.share_memory()
    
    workers=[Worker(0,global_model,optimizer,device,args,True)]
    test_worker=[mp.Process(target=test,args=(global_model,device,args))]
    
    workers=workers+[Worker(i,global_model,optimizer,device,args) for i in range(1,NUM_PROCESSES)]
    workers=workers+test_worker
    [w.start() for w in workers]
    
    

    #workers.append(test_worker)
    
    [w.join() for w in workers]
    
    
    

    
    
    